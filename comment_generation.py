import argparse
from collections import Counter
import json
import numpy as np
import os
import random
import sys
import torch
from torch import nn
from typing import List, NamedTuple

from dpu_utils.mlutils import Vocabulary

from constants import *
from data_utils import read_full_examples_from_file, read_examples_from_file, GenerationBatchData
from generation_decoder import GenerationDecoder
from embedding_store import EmbeddingStore
from encoder import Encoder
from eval_utils import compute_accuracy, compute_bleu, compute_meteor
from tensor_utils import get_invalid_copy_locations

class CommentGenerationModel(nn.Module):
    """Simple Seq2Seq w/ attention + copy model for pure comment generation (i.e. generating a comment given a method)."""
    def __init__(self, model_path):
        super(CommentGenerationModel, self).__init__()
        self.model_path = model_path
        self.torch_device_name = 'cpu'
    
    def initialize(self, train_data, embedding_store=None):
        """Initializes model parameters from pre-defined hyperparameters and other hyperparameters
           that are computed based on statistics over the training data."""
        nl_lengths = []
        code_lengths = []
        nl_token_counter = Counter()
        code_token_counter = Counter()

        for ex in train_data:
            trg_sequence = [START] + ex.new_comment_tokens + [END]
            nl_token_counter.update(trg_sequence)
            nl_lengths.append(len(trg_sequence))

            code_sequence = ex.new_code_tokens
            code_token_counter.update(code_sequence)
            code_lengths.append(len(code_sequence))
        
        self.max_nl_length = int(np.percentile(np.asarray(sorted(nl_lengths)),
            LENGTH_CUTOFF_PCT))
        self.max_code_length = int(np.percentile(np.asarray(sorted(code_lengths)),
            LENGTH_CUTOFF_PCT))
    
        nl_counts = np.asarray(sorted(nl_token_counter.values()))
        nl_threshold = int(np.percentile(nl_counts, VOCAB_CUTOFF_PCT)) + 1
        code_counts = np.asarray(sorted(code_token_counter.values()))
        code_threshold = int(np.percentile(nl_counts, VOCAB_CUTOFF_PCT)) + 1
        
        if embedding_store is None:
            self.embedding_store = EmbeddingStore(nl_threshold, NL_EMBEDDING_SIZE, nl_token_counter,
                code_threshold, CODE_EMBEDDING_SIZE, code_token_counter, DROPOUT_RATE)
        else:
            self.embedding_store = embedding_store
        
        self.code_encoder = Encoder(CODE_EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_RATE)
        self.decoder = GenerationDecoder(NL_EMBEDDING_SIZE, DECODER_HIDDEN_SIZE,
            2*HIDDEN_SIZE, self.embedding_store, NL_EMBEDDING_SIZE, DROPOUT_RATE)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LR)
    
    def get_batches(self, dataset, shuffle=False):
        """Divides the dataset into batches based on pre-defined BATCH_SIZE hyperparameter.
           Each batch is tensorized so that it can be directly passed into the network."""
        batches = []
        if shuffle:
            random.shuffle(dataset)
        
        curr_idx = 0
        while curr_idx < len(dataset):
            start_idx = curr_idx
            end_idx = min(start_idx + BATCH_SIZE, len(dataset))
            
            code_token_ids = []
            code_lengths = []
            trg_token_ids = []
            trg_extended_token_ids = []
            trg_lengths = []
            invalid_copy_positions = []
            inp_str_reps = []
            inp_ids = []

            for i in range(start_idx, end_idx):
                code_sequence = dataset[i].new_code_tokens
                code_sequence_ids = self.embedding_store.get_padded_code_ids(
                    code_sequence, self.max_code_length)
                code_length = min(len(code_sequence), self.max_code_length)
                code_token_ids.append(code_sequence_ids)
                code_lengths.append(code_length)
                
                ex_inp_str_reps = []
                ex_inp_ids = []
                
                extra_counter = len(self.embedding_store.nl_vocabulary)
                max_limit = len(self.embedding_store.nl_vocabulary) + self.max_code_length
                out_ids = set()
                
                for c in code_sequence[:code_length]:
                    nl_id = self.embedding_store.get_nl_id(c)
                    if self.embedding_store.is_nl_unk(nl_id) and extra_counter < max_limit:
                        if c in ex_inp_str_reps:
                            nl_id = ex_inp_ids[ex_inp_str_reps.index(c)]
                        else:
                            nl_id = extra_counter
                            extra_counter += 1

                    out_ids.add(nl_id)
                    ex_inp_str_reps.append(c)
                    ex_inp_ids.append(nl_id)
                
                trg_sequence = [START] + dataset[i].new_comment_tokens + [END]
                trg_sequence_ids = self.embedding_store.get_padded_nl_ids(
                    trg_sequence, self.max_nl_length)
                trg_extended_sequence_ids = self.embedding_store.get_extended_padded_nl_ids(
                    trg_sequence, self.max_nl_length, ex_inp_ids, ex_inp_str_reps)
                
                trg_token_ids.append(trg_sequence_ids)
                trg_extended_token_ids.append(trg_extended_sequence_ids)
                trg_lengths.append(min(len(trg_sequence), self.max_nl_length))
                inp_str_reps.append(ex_inp_str_reps)
                inp_ids.append(ex_inp_ids)

                invalid_copy_positions.append(get_invalid_copy_locations(ex_inp_str_reps, self.max_code_length,
                    trg_sequence, self.max_nl_length))

            batches.append(GenerationBatchData(torch.tensor(code_token_ids, dtype=torch.int64, device=self.get_device()),
                                               torch.tensor(code_lengths, dtype=torch.int64, device=self.get_device()),
                                               torch.tensor(trg_token_ids, dtype=torch.int64, device=self.get_device()),
                                               torch.tensor(trg_extended_token_ids, dtype=torch.int64, device=self.get_device()),
                                               torch.tensor(trg_lengths, dtype=torch.int64, device=self.get_device()),
                                               torch.tensor(invalid_copy_positions, dtype=torch.uint8, device=self.get_device()),
                                               inp_str_reps, inp_ids))
            curr_idx = end_idx
        return batches
    
    def get_encoder_output(self, batch_data):
        """Gets hidden states, final state, and a length-mask from the encoder."""
        code_embedded_tokens = self.embedding_store.get_code_embeddings(batch_data.code_ids)
        code_hidden_states, code_final_state = self.code_encoder.forward(code_embedded_tokens,
            batch_data.code_lengths, self.get_device())
        mask = (torch.arange(
            code_hidden_states.shape[1], device=self.get_device()).view(1, -1) >= batch_data.code_lengths.view(-1, 1)).unsqueeze(1)
        return code_hidden_states, code_final_state, mask
        
    def forward(self, batch_data):
        """Computes the loss against the gold sequences corresponding to the examples in the batch. NOTE: teacher-forcing."""
        encoder_hidden_states, initial_state, inp_length_mask = self.get_encoder_output(batch_data)
        decoder_input_embeddings = self.embedding_store.get_nl_embeddings(batch_data.trg_nl_ids)[:, :-1]
        decoder_states, decoder_final_state, generation_logprobs, copy_logprobs = self.decoder.forward(
            initial_state, decoder_input_embeddings, encoder_hidden_states, inp_length_mask)
        
        gold_generation_ids = batch_data.trg_nl_ids[:, 1:].unsqueeze(-1)
        gold_generation_logprobs = torch.gather(input=generation_logprobs, dim=-1,
                                                index=gold_generation_ids).squeeze(-1)
        copy_logprobs = copy_logprobs.masked_fill(
            batch_data.invalid_copy_positions[:,1:,:encoder_hidden_states.shape[1]], float('-inf'))
        gold_copy_logprobs = copy_logprobs.logsumexp(dim=-1)

        gold_logprobs = torch.logsumexp(torch.cat(
            [gold_generation_logprobs.unsqueeze(-1), gold_copy_logprobs.unsqueeze(-1)], dim=-1), dim=-1)
        gold_logprobs = gold_logprobs.masked_fill(torch.arange(batch_data.trg_nl_ids[:,1:].shape[-1],
            device=self.get_device()).unsqueeze(0) >= batch_data.trg_nl_lengths.unsqueeze(-1)-1, 0)
        
        likelihood_by_example = gold_logprobs.sum(dim=-1)

        # Normalizing by length. Seems to help
        likelihood_by_example = likelihood_by_example/(batch_data.trg_nl_lengths-1).float() 
        
        return -(likelihood_by_example).mean()
        
    def compute_generation_likelihood(self, batch_data):
        """This is not used by the generation model but rather the comment update model for re-ranking. It computes P(Comment|Method)."""
        with torch.no_grad():
            encoder_hidden_states, initial_state, inp_length_mask = self.get_encoder_output(batch_data)
            decoder_input_embeddings = self.embedding_store.get_nl_embeddings(batch_data.trg_nl_ids)[:, :-1]
            decoder_states, decoder_final_state, generation_logprobs, copy_logprobs = self.decoder.forward(initial_state,
                decoder_input_embeddings, encoder_hidden_states, inp_length_mask)
            
            gold_generation_ids = batch_data.trg_nl_ids[:, 1:].unsqueeze(-1)
            gold_generation_logprobs = torch.gather(input=generation_logprobs, dim=-1,
                                                    index=gold_generation_ids).squeeze(-1)
            copy_logprobs = copy_logprobs.masked_fill(
                batch_data.invalid_copy_positions[:,1:,:encoder_hidden_states.shape[1]], float('-inf'))
            gold_copy_logprobs = copy_logprobs.logsumexp(dim=-1)

            gold_logprobs = torch.logsumexp(torch.cat(
                [gold_generation_logprobs.unsqueeze(-1), gold_copy_logprobs.unsqueeze(-1)], dim=-1), dim=-1)
            gold_logprobs = gold_logprobs.masked_fill(torch.arange(batch_data.trg_nl_ids[:,1:].shape[-1],
                device=self.get_device()).unsqueeze(0) >= batch_data.trg_nl_lengths.unsqueeze(-1)-1, 0)
            return torch.exp(gold_logprobs.sum(dim=-1)/(batch_data.trg_nl_lengths-1).float())
        
    def greedy_decode(self, batch_data):
        """Predicts a comment for every method in the batch in a greedy manner."""
        encoder_hidden_states, initial_state, inp_length_mask = self.get_encoder_output(batch_data)
        predictions, scores = self.decoder.greedy_decode(initial_state, encoder_hidden_states,
            inp_length_mask, self.max_nl_length, batch_data, self.get_device())
        
        batch_size = initial_state.shape[0]

        decoded_tokens = []
        for i in range(batch_size):
            token_ids = predictions[i]
            tokens = self.embedding_store.get_nl_tokens(token_ids, batch_data.input_ids[i],
                batch_data.input_str_reps[i])
            decoded_tokens.append(tokens)
        return decoded_tokens
    
    def get_device(self):
        """Returns the proper device."""
        if self.torch_device_name == 'gpu':
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def run_gradient_step(self, batch_data):
        """Performs gradient step."""
        self.optimizer.zero_grad()
        loss = self.forward(batch_data)
        loss.backward()
        self.optimizer.step()
        return float(loss.cpu())

    def run_train(self, train_data, valid_data):
        """Runs training over the entire training set across several epochs. Following each epoch,
           loss on the validation data is computed. If the validation loss has improved, save the model.
           Early-stopping is employed to stop training if validation hasn't improved for a certain number
           of epochs."""
        valid_batches = self.get_batches(valid_data)
        train_batches = self.get_batches(train_data, shuffle=True)

        best_loss = float('inf')
        patience_tally = 0

        for epoch in range(MAX_EPOCHS):
            if patience_tally > PATIENCE:
                print('Terminating')
                break
            
            self.train()
            random.shuffle(train_batches)
            
            train_loss = 0
            for batch_data in train_batches:
                train_loss += self.run_gradient_step(batch_data)
        
            self.eval()
            validation_loss = 0
            with torch.no_grad():
                for batch_data in valid_batches:
                    validation_loss += float(
                        self.forward(batch_data).cpu())

            validation_loss = validation_loss/len(valid_batches)

            if validation_loss <= best_loss:
                torch.save(self, self.model_path)
                saved = True
                best_loss = validation_loss
                patience_tally = 0
            else:
                saved = False
                patience_tally += 1
            
            print('Epoch: {}'.format(epoch))
            print('Training loss: {}'.format(train_loss/len(train_batches)))
            print('Validation loss: {}'.format(validation_loss))
            if saved:
                print('Saved')
            print('-----------------------------------')
    
    def run_evaluation(self, test_data):
        """Generates predicted comments for all comments in the test set and computes evaluation metrics."""
        self.eval()

        test_batches = self.get_batches(test_data)
        test_predictions = []

        # Lists of string predictions
        gold_strs = []
        pred_strs = []
        
        # Lists of tokenized predictions
        references = []
        pred_instances = []

        with torch.no_grad():
            for b_idx, batch_data in enumerate(test_batches):
                test_predictions.extend(self.greedy_decode(batch_data))
            
        for i in range(len(test_predictions)):
            prediction = test_predictions[i]
            gold_str = test_data[i].new_comment
            pred_str = ' '.join(prediction)
            
            gold_strs.append(gold_str)
            pred_strs.append(pred_str)
            
            references.append([test_data[i].new_comment_tokens])
            pred_instances.append(prediction)

            print('Gold comment: {}'.format(gold_str))
            print('Predicted comment: {}'.format(pred_str))
            print('----------------------------')
        
        predicted_accuracy = compute_accuracy(gold_strs, pred_strs)
        predicted_bleu = compute_bleu(references, pred_instances)
        predicted_meteor = compute_meteor(references, pred_instances)

        print('Predicted Accuracy: {}'.format(predicted_accuracy))
        print('Predicted BLEU: {}'.format(predicted_bleu))
        print('Predicted Meteor: {}\n'.format(predicted_meteor))
    
def write_embeddings(model, nl_embedding_file, code_embedding_file):
    """Saves embeddings from pre-trained model onto disk."""
    nl_embeddings = dict()
    code_embeddings = dict()
    
    with torch.no_grad():
        nl_vocab = model.embedding_store.nl_vocabulary
        nl_weights = model.embedding_store.nl_embedding_layer.weight
        code_vocab = model.embedding_store.code_vocabulary
        code_weights = model.embedding_store.code_embedding_layer.weight
        for i, nl_word in enumerate(nl_vocab.id_to_token):
            nl_embeddings[nl_word] = list(nl_weights[i].detach().numpy())
            nl_embeddings[nl_word] = [float(f) for f in nl_embeddings[nl_word]]
        
        for i, code_word in enumerate(code_vocab.id_to_token):
            code_embeddings[code_word] = list(code_weights[i].detach().numpy())
            code_embeddings[code_word] = [float(f) for f in code_embeddings[code_word]]
    
    with open(nl_embedding_file, 'w+') as f:
        json.dump(nl_embeddings, f)
    
    with open(code_embedding_file, 'w+') as f:
        json.dump(code_embeddings, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path')
    parser.add_argument('-model_path')
    parser.add_argument('--test_mode', action='store_true')
    parser.add_argument('--get_embeddings', action='store_true')
    args = parser.parse_args()

    if args.test_mode:
        model = torch.load(args.model_path)
        model.torch_device_name = 'cpu'
        model.cpu()
        for c in model.children():
            c.cpu()
        
        if args.get_embeddings:
            nl_embedding_file = os.path.join('embeddings', 'nl_embeddings.json')
            code_embedding_file = os.path.join('embeddings', 'code_embeddings.json')
            write_embeddings(model, nl_embedding_file, code_embedding_file)
        else:
            test_examples = read_full_examples_from_file(os.path.join(args.data_path, 'test.json'))
            model.run_evaluation(test_examples)
    else:
        train_examples = read_full_examples_from_file(os.path.join(args.data_path, 'train.json'))
        valid_examples = read_full_examples_from_file(os.path.join(args.data_path, 'valid.json'))

        model = CommentGenerationModel(args.model_path)
        model.initialize(train_examples)
        if torch.cuda.is_available():
            model.torch_device_name = 'gpu'
            model.cuda()
            for c in model.children():
                c.cuda()
        else:
            model.torch_device_name = 'cpu'
            model.cpu()
            for c in model.children():
                c.cpu()
        
        model.run_train(train_examples, valid_examples)
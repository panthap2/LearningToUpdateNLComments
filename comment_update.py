import argparse
from collections import Counter
import numpy as np
import os
import random
import sys
import torch
from torch import nn

from dpu_utils.mlutils import Vocabulary

import comment_generation
from comment_generation import CommentGenerationModel
from constants import *
from data_utils import read_examples_from_file, UpdateBatchData, Example
import diff_utils
from embedding_store import EmbeddingStore
from encoder import Encoder
from external_cache import get_code_features,get_nl_features, NUM_CODE_FEATURES, NUM_NL_FEATURES,\
    get_old_code, get_new_code
from eval_utils import compute_accuracy, compute_bleu, compute_meteor, write_predictions,\
    compute_sentence_bleu, compute_sentence_meteor, compute_sari, compute_gleu
from tensor_utils import merge_encoder_outputs, get_invalid_copy_locations
from update_decoder import UpdateDecoder

class CommentUpdateModel(nn.Module):
    """Edit model which learns to map a sequence of code edits to a sequence of comment edits and then applies the edits to the
       old comment in order to produce an updated comment."""
    def __init__(self, model_path):
        super(CommentUpdateModel, self).__init__()
        self.model_path = model_path
        self.torch_device_name = 'cpu'
    
    def initialize(self, train_data):
        """Initializes model parameters from pre-defined hyperparameters and other hyperparameters
           that are computed based on statistics over the training data."""
        nl_lengths = []
        code_lengths = []
        nl_token_counter = Counter()
        code_token_counter = Counter()

        for ex in train_data:
            trg_sequence = [START] + ex.span_minimal_diff_comment_tokens + [END]
            nl_token_counter.update(trg_sequence)
            nl_lengths.append(len(trg_sequence))

            old_nl_sequence = ex.old_comment_tokens
            nl_token_counter.update(old_nl_sequence)
            nl_lengths.append(len(old_nl_sequence))

            code_sequence = ex.span_diff_code_tokens
            code_token_counter.update(code_sequence)
            code_lengths.append(len(code_sequence))
        
        self.max_nl_length = int(np.percentile(np.asarray(sorted(nl_lengths)),
            LENGTH_CUTOFF_PCT))
        self.max_code_length = int(np.percentile(np.asarray(sorted(code_lengths)),
            LENGTH_CUTOFF_PCT))
        self.max_vocab_extension = self.max_nl_length + self.max_code_length
    
        nl_counts = np.asarray(sorted(nl_token_counter.values()))
        nl_threshold = int(np.percentile(nl_counts, VOCAB_CUTOFF_PCT)) + 1
        code_counts = np.asarray(sorted(code_token_counter.values()))
        code_threshold = int(np.percentile(nl_counts, VOCAB_CUTOFF_PCT)) + 1

        self.embedding_store = EmbeddingStore(nl_threshold, NL_EMBEDDING_SIZE, nl_token_counter,
            code_threshold, CODE_EMBEDDING_SIZE, code_token_counter,
            DROPOUT_RATE, True)
        
        self.code_encoder = Encoder(CODE_EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_RATE)
        self.nl_encoder = Encoder(NL_EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_RATE)
        self.decoder = UpdateDecoder(NL_EMBEDDING_SIZE, DECODER_HIDDEN_SIZE,
            2*HIDDEN_SIZE, self.embedding_store, NL_EMBEDDING_SIZE, DROPOUT_RATE)
        self.encoder_final_to_decoder_initial = nn.Parameter(torch.randn(2*NUM_ENCODERS*HIDDEN_SIZE,
            DECODER_HIDDEN_SIZE, dtype=torch.float, requires_grad=True))
        
        self.code_features_to_embedding = nn.Linear(CODE_EMBEDDING_SIZE + NUM_CODE_FEATURES,
            CODE_EMBEDDING_SIZE, bias=False)
        self.nl_features_to_embedding = nn.Linear(
            NL_EMBEDDING_SIZE + NUM_NL_FEATURES,
            NL_EMBEDDING_SIZE, bias=False)
        
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
            old_nl_token_ids = []
            old_nl_lengths = []
            trg_token_ids = []
            trg_extended_token_ids = []
            trg_lengths = []
            invalid_copy_positions = []
            inp_str_reps = []
            inp_ids = []
            code_features = []
            nl_features = []

            for i in range(start_idx, end_idx):
                code_sequence = dataset[i].span_diff_code_tokens
                code_sequence_ids = self.embedding_store.get_padded_code_ids(
                    code_sequence, self.max_code_length)
                code_length = min(len(code_sequence), self.max_code_length)
                code_token_ids.append(code_sequence_ids)
                code_lengths.append(code_length)

                old_nl_sequence = dataset[i].old_comment_tokens
                old_nl_length = min(len(old_nl_sequence), self.max_nl_length)
                old_nl_sequence_ids = self.embedding_store.get_padded_nl_ids(
                    old_nl_sequence, self.max_nl_length)
                
                old_nl_token_ids.append(old_nl_sequence_ids)
                old_nl_lengths.append(old_nl_length)
                
                ex_inp_str_reps = []
                ex_inp_ids = []
                
                extra_counter = len(self.embedding_store.nl_vocabulary)
                max_limit = len(self.embedding_store.nl_vocabulary) + self.max_vocab_extension
                out_ids = set()

                copy_inputs = code_sequence[:code_length] + old_nl_sequence[:old_nl_length]
                for c in copy_inputs:
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
                
                trg_sequence = trg_sequence = [START] + dataset[i].span_minimal_diff_comment_tokens + [END]
                trg_sequence_ids = self.embedding_store.get_padded_nl_ids(
                    trg_sequence, self.max_nl_length)
                trg_extended_sequence_ids = self.embedding_store.get_extended_padded_nl_ids(
                    trg_sequence, self.max_nl_length, ex_inp_ids, ex_inp_str_reps)
                
                trg_token_ids.append(trg_sequence_ids)
                trg_extended_token_ids.append(trg_extended_sequence_ids)
                trg_lengths.append(min(len(trg_sequence), self.max_nl_length))
                inp_str_reps.append(ex_inp_str_reps)
                inp_ids.append(ex_inp_ids)

                invalid_copy_positions.append(get_invalid_copy_locations(ex_inp_str_reps, self.max_vocab_extension,
                    trg_sequence, self.max_nl_length))
                code_features.append(get_code_features(code_sequence, dataset[i], self.max_code_length))
                nl_features.append(get_nl_features(old_nl_sequence, dataset[i], self.max_nl_length))
                
            batches.append(UpdateBatchData(torch.tensor(code_token_ids, dtype=torch.int64, device=self.get_device()),
                                           torch.tensor(code_lengths, dtype=torch.int64, device=self.get_device()),
                                           torch.tensor(old_nl_token_ids, dtype=torch.int64, device=self.get_device()),
                                           torch.tensor(old_nl_lengths, dtype=torch.int64, device=self.get_device()),
                                           torch.tensor(trg_token_ids, dtype=torch.int64, device=self.get_device()),
                                           torch.tensor(trg_extended_token_ids, dtype=torch.int64, device=self.get_device()),
                                           torch.tensor(trg_lengths, dtype=torch.int64, device=self.get_device()),
                                           torch.tensor(invalid_copy_positions, dtype=torch.uint8, device=self.get_device()),
                                           inp_str_reps, inp_ids, torch.tensor(code_features, dtype=torch.float32, device=self.get_device()),
                                           torch.tensor(nl_features, dtype=torch.float32, device=self.get_device())))
            curr_idx = end_idx
        return batches
    
    def get_encoder_output(self, batch_data):
        """Gets hidden states, final state, and a length masks corresponding to each encoder."""
        code_embedded_tokens = self.code_features_to_embedding(torch.cat(
            [self.embedding_store.get_code_embeddings(batch_data.code_ids), batch_data.code_features], dim=-1))
        code_hidden_states, code_final_state = self.code_encoder.forward(code_embedded_tokens,
            batch_data.code_lengths, self.get_device())
        
        old_nl_embedded_tokens = self.nl_features_to_embedding(torch.cat(
            [self.embedding_store.get_nl_embeddings(batch_data.old_nl_ids), batch_data.nl_features], dim=-1))
        old_nl_hidden_states, old_nl_final_state = self.nl_encoder.forward(old_nl_embedded_tokens,
            batch_data.old_nl_lengths, self.get_device())
        
        encoder_hidden_states, input_lengths = merge_encoder_outputs(code_hidden_states,
            batch_data.code_lengths, old_nl_hidden_states, batch_data.old_nl_lengths, self.get_device())

        encoder_final_state = torch.einsum('bd,dh->bh',
            torch.cat([code_final_state, old_nl_final_state], dim=-1),
            self.encoder_final_to_decoder_initial)
        mask = (torch.arange(
            encoder_hidden_states.shape[1], device=self.get_device()).view(1, -1) >= input_lengths.view(-1, 1)).unsqueeze(1)
        
        code_masks = (torch.arange(
            code_hidden_states.shape[1], device=self.get_device()).view(1, -1) >= batch_data.code_lengths.view(-1, 1)).unsqueeze(1)
        old_nl_masks = (torch.arange(
            old_nl_hidden_states.shape[1], device=self.get_device()).view(1, -1) >= batch_data.old_nl_lengths.view(-1, 1)).unsqueeze(1)
        
        return encoder_hidden_states, encoder_final_state, mask, code_hidden_states, old_nl_hidden_states, code_masks, old_nl_masks
    
    def forward(self, batch_data):
        """Computes the loss against the gold sequences corresponding to the examples in the batch. NOTE: teacher-forcing."""
        encoder_hidden_states, initial_state, inp_length_mask, code_hidden_states, old_nl_hidden_states, code_masks, old_nl_masks = self.get_encoder_output(batch_data)
        decoder_input_embeddings = self.embedding_store.get_nl_embeddings(batch_data.trg_nl_ids)[:, :-1]
        decoder_states, decoder_final_state, generation_logprobs, copy_logprobs = self.decoder.forward(initial_state, decoder_input_embeddings,
            encoder_hidden_states, code_hidden_states, old_nl_hidden_states, inp_length_mask, code_masks, old_nl_masks)

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
    
    def beam_decode(self, batch_data):
        """Performs beam search on the decoder to get candidate predictions for every example in the batch."""
        encoder_hidden_states, initial_state, inp_length_mask, code_hidden_states, old_nl_hidden_states, code_masks, old_nl_masks = self.get_encoder_output(batch_data)
        predictions, scores = self.decoder.beam_decode(initial_state, encoder_hidden_states,
            code_hidden_states, old_nl_hidden_states, inp_length_mask, self.max_nl_length, batch_data, code_masks, old_nl_masks, self.get_device())
        
        decoded_output = []
        batch_size = initial_state.shape[0]

        for i in range(batch_size):
            beam_output = []
            for j in range(len(predictions[i])):
                token_ids = predictions[i][j]
                tokens = self.embedding_store.get_nl_tokens(token_ids, batch_data.input_ids[i],
                    batch_data.input_str_reps[i])
                beam_output.append((tokens, scores[i][j]))
            decoded_output.append(beam_output)
        return decoded_output
    
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
    
    def get_likelihood_scores(self, comment_generation_model, formatted_beam_predictions, test_example):
        """Computes the generation likelihood score for each beam prediction based on the pre-trained
           comment generation model."""
        batch_examples = []
        for j in range(len(formatted_beam_predictions)):
            batch_examples.append(Example(test_example.id, test_example.old_comment, test_example.old_comment_tokens,
                ' '.join(formatted_beam_predictions[j]), formatted_beam_predictions[j],
                test_example.old_code, test_example.old_code_tokens, test_example.new_code,
                test_example.new_code_tokens))
        
        batch_data = comment_generation_model.get_batches(batch_examples)[0]
        return np.asarray(comment_generation_model.compute_generation_likelihood(batch_data))
    
    def get_generation_model(self):
        """Loads the pre-trained comment generation model needed for re-ranking.
           NOTE: the path is hard-coded here so may need to be modified."""
        comment_generation_model = torch.load(FULL_GENERATION_MODEL_PATH)
        comment_generation_model.torch_device_name = 'cpu'
        comment_generation_model.cpu()
        for c in comment_generation_model.children():
            c.cpu()
        comment_generation_model.eval()
        return comment_generation_model

    def run_evaluation(self, test_data, rerank):
        """Predicts updated comments for all comments in the test set and computes evaluation metrics."""
        self.eval()

        test_batches = self.get_batches(test_data)
        test_predictions = []
        generation_predictions = []

        gold_strs = []
        pred_strs = []
        src_strs = []

        references = []
        pred_instances = []

        with torch.no_grad():
            for b_idx, batch_data in enumerate(test_batches):
                test_predictions.extend(self.beam_decode(batch_data))

        if not rerank:
            test_predictions = [pred[0][0] for pred in test_predictions]
        else:
            comment_generation_model = self.get_generation_model()
            with torch.no_grad():
                generation_test_batches = comment_generation_model.get_batches(test_data)
                for gen_batch_data in generation_test_batches:
                    generation_predictions.extend(
                        comment_generation_model.greedy_decode(gen_batch_data))
            
            reranked_predictions = []
            for i in range(len(test_predictions)):
                formatted_beam_predictions = []
                model_scores = np.zeros(len(test_predictions[i]), dtype=np.float)
                generated = generation_predictions[i]
                old_comment_tokens = test_data[i].old_comment_tokens
                
                for b, (b_pred, b_score) in enumerate(test_predictions[i]):
                    b_pred_str = diff_utils.format_minimal_diff_spans(test_data[i].old_comment_tokens, b_pred)
                    formatted_beam_predictions.append(b_pred_str.split(' '))
                    model_scores[b] = b_score
                
                likelihood_scores = self.get_likelihood_scores(comment_generation_model,
                    formatted_beam_predictions, test_data[i])
                old_meteor_scores = compute_sentence_meteor(
                        [[old_comment_tokens] for _ in range(len(formatted_beam_predictions))],
                        formatted_beam_predictions)
                
                rerank_scores = [(model_scores[j] * MODEL_LAMBDA) + (likelihood_scores[j] * LIKELIHOOD_LAMBDA) + (
                        old_meteor_scores[j] * OLD_METEOR_LAMBDA) for j in range(len(formatted_beam_predictions))]
                
                sorted_indices = np.argsort(-np.asarray(rerank_scores))
                reranked_predictions.append(test_predictions[i][sorted_indices[0]][0])

            test_predictions = reranked_predictions

        for i in range(len(test_predictions)):
            pred_str = diff_utils.format_minimal_diff_spans(test_data[i].old_comment_tokens, test_predictions[i])
            prediction = pred_str.split()
            gold_str = test_data[i].new_comment
            
            gold_strs.append(gold_str)
            pred_strs.append(pred_str)
            src_strs.append(test_data[i].old_comment)
            
            references.append([test_data[i].new_comment_tokens])
            pred_instances.append(prediction)

            print('Old comment: {}'.format(test_data[i].old_comment))
            print('Gold comment: {}'.format(gold_str))
            print('Predicted comment: {}'.format(pred_str))
            print('Raw prediction: {}\n'.format(' '.join(test_predictions[i])))
            try:
                print('Old code:\n{}\n'.format(get_old_code(test_data[i])))
            except:
                print('Failed to print old code\n')
            print('New code:\n{}\n'.format(get_new_code(test_data[i])))
            print('----------------------------')

        if rerank:
            prediction_file = '{}_rerank.txt'.format(self.model_path.split('.')[0])
        else:
            prediction_file = '{}.txt'.format(self.model_path.split('.')[0])
        
        write_predictions(pred_strs, prediction_file)
        write_predictions(src_strs, 'src.txt')
        write_predictions(gold_strs, 'ref.txt')

        predicted_accuracy = compute_accuracy(gold_strs, pred_strs)
        predicted_bleu = compute_bleu(references, pred_instances)
        predicted_meteor = compute_meteor(references, pred_instances)
        predicted_sari = compute_sari(test_data, pred_instances)
        predicted_gleu = compute_gleu(test_data, 'src.txt', 'ref.txt', prediction_file)

        print('Predicted Accuracy: {}'.format(predicted_accuracy))
        print('Predicted BLEU: {}'.format(predicted_bleu))
        print('Predicted Meteor: {}'.format(predicted_meteor))
        print('Predicted SARI: {}'.format(predicted_sari))
        print('Predicted GLEU: {}\n'.format(predicted_gleu))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path')
    parser.add_argument('-model_path')
    parser.add_argument('--test_mode', action='store_true')
    parser.add_argument('--rerank', action='store_true')
    args = parser.parse_args()

    if args.test_mode:
        test_examples = read_examples_from_file(os.path.join(args.data_path, 'test.json'))
        
        model = torch.load(args.model_path)
        model.torch_device_name = 'cpu'
        model.cpu()
        for c in model.children():
            c.cpu()
        
        model.run_evaluation(test_examples, args.rerank)

    else:
        train_examples = read_examples_from_file(os.path.join(args.data_path, 'train.json'))
        valid_examples = read_examples_from_file(os.path.join(args.data_path, 'valid.json'))

        model = CommentUpdateModel(args.model_path)
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

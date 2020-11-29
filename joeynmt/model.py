# coding: utf-8
"""
Module to represents whole models
"""
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from joeynmt.initialization import initialize_model
from joeynmt.embeddings import Embeddings
from joeynmt.encoders import Encoder, RecurrentEncoder, TransformerEncoder
from joeynmt.decoders import Decoder, RecurrentDecoder, TransformerDecoder, CriticDecoder, CriticTransformerDecoder
from joeynmt.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
from joeynmt.vocabulary import Vocabulary
from torch.distributions import Categorical
from joeynmt.batch import Batch
from joeynmt.helpers import ConfigurationError, tensor_to_float, log_peakiness, join_strings
from joeynmt.metrics import bleu
import threading


class RewardRegressionModel(nn.Module):
    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.l1 = nn.Linear(D_in, H)
        self.relu = nn.ReLU()
        self.l2=nn.Linear(H, D_out)

    def forward(self, X):
        return self.l2(self.relu(self.l1(X)))

class Model(nn.Module):
    """
    Base Model class
    """

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: Embeddings,
                 trg_embed: Embeddings,
                 src_vocab: Vocabulary,
                 trg_vocab: Vocabulary) -> None:
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param src_embed: source embedding
        :param trg_embed: target embedding
        :param src_vocab: source vocabulary
        :param trg_vocab: target vocabulary
        """
        super().__init__()

        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.encoder = encoder
        self.decoder = decoder
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.bos_index = self.trg_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.trg_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.trg_vocab.stoi[EOS_TOKEN]
        self._loss_function = None # set by the TrainManager
        self.bleu = []
        self.counter = 0 

    @property
    def loss_function(self):
        return self._x

    @loss_function.setter
    def loss_function(self, loss_function: Callable):
        self._loss_function = loss_function

    def measure_transformer_peakiness(self, max_output_length, src: Tensor, trg: Tensor, src_mask: Tensor,
                        src_length: Tensor) -> Tensor:
        encoder_output, encoder_hidden = self._encode(
            src, src_length,
            src_mask)
        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = int(max(src_length.cpu().numpy()) * 1.5)
        batch_size = src_mask.size(0)
        ys = encoder_output.new_full([batch_size, 1], self.bos_index, dtype=torch.long)
        trg_mask = src_mask.new_ones([1, 1, 1])
        log_probs = 0
        distributions = []

        finished = src_mask.new_zeros((batch_size)).byte()
        for i in range(max_output_length):
            logits, out, _, _ = self.decoder(
                trg_embed=self.trg_embed(ys),
                encoder_output=encoder_output,
                encoder_hidden=None,
                src_mask=src_mask,
                unroll_steps=None,
                hidden=None,
                trg_mask=trg_mask
            )
            logits = logits[:, -1]
            distrib = Categorical(logits=logits)
            distributions.append(distrib)
            probabilities = distrib.probs
            
            top100_probs, top100_probs_probs_index = probabilities.topk(100, largest=True, sorted=True)
            word = distrib.sample()
            next_word = word
            log_probs -= distrib.log_prob(next_word)
            ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)

            # check if previous symbol was <eos>
            is_eos = torch.eq(next_word, self.eos_index)
            finished += is_eos
            # stop predicting if <eos> reached for all elements in batch
            if (finished >= 1).sum() == batch_size:
                break

        ys = ys[:, 1:]
        predicted_output = self.trg_vocab.arrays_to_sentences(arrays=ys,
                                                        cut_at_eos=True)
        gold_output = self.trg_vocab.arrays_to_sentences(arrays=trg,
                                                    cut_at_eos=True)
        predicted_strings = [self.join_strings(wordlist) for wordlist in predicted_output]
        gold_strings = [self.join_strings(wordlist) for wordlist in gold_output]
        #batch_loss, rewards, old_bleus = self.loss_function(predicted_strings, gold_strings, log_probs, top_words, ys, trg)
        return _, [entropy/max_output_length, gold_strings, predicted_strings, sentence_highest_words, sentence_probability, sentence_highest_word, sentence_highest_probability, gold_probabilities, gold_token_ranks, [], []]

   
    def reinforce_transformer(self, max_output_length, src: Tensor, trg: Tensor, src_mask: Tensor,
                        src_length: Tensor, temperature: float) -> Tensor:
        encoder_output, encoder_hidden = self._encode(
            src, src_length,
            src_mask)
        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = int(max(src_length.cpu().numpy()) * 1.5)
        batch_size = src_mask.size(0)
        ys = encoder_output.new_full([batch_size, 1], self.bos_index, dtype=torch.long)
        trg_mask = src_mask.new_ones([1, 1, 1])
        distributions = []
        log_probs = 0
        #log_probs = torch.FloatTensor([0]*batch_size)
        #log_probs.requires_grad=True
        finished = src_mask.new_zeros((batch_size)).byte()
        for i in range(max_output_length):
            logits, out, _, _ = self.decoder(
                trg_embed=self.trg_embed(ys),
                encoder_output=encoder_output,
                encoder_hidden=None,
                src_mask=src_mask,
                unroll_steps=None,
                hidden=None,
                trg_mask=trg_mask
            )
            logits = logits[:, -1]/temperature
            distrib = Categorical(logits=logits)
            distributions.append(distrib)
            word = distrib.sample()
            next_word = word

            #current_log_probs = distrib.log_prob(next_word)
            #with torch.no_grad():
            #    for index in range(len(next_word)):
            #        if not torch.eq(next_word[index], self.eos_index) and not torch.eq(next_word[index], self.pad_index):
            #            log_probs[index] -= current_log_probs[index]
            log_probs -= distrib.log_prob(next_word)
            ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)

            # check if previous symbol was <eos>
            is_eos = torch.eq(next_word, self.eos_index)
            finished += is_eos
            # stop predicting if <eos> reached for all elements in batch
            if (finished >= 1).sum() == batch_size:
                break

        ys = ys[:, 1:]
        predicted_output = self.trg_vocab.arrays_to_sentences(arrays=ys,
                                                        cut_at_eos=True)
        gold_output = self.trg_vocab.arrays_to_sentences(arrays=trg,
                                                    cut_at_eos=True)
        predicted_strings = [join_strings(wordlist) for wordlist in predicted_output]
        gold_strings = [join_strings(wordlist) for wordlist in gold_output]
        batch_loss, rewards, old_bleus = self.loss_function(predicted_strings, gold_strings,  log_probs, ys, trg)
        return batch_loss, log_peakiness(self.pad_index, self.trg_vocab, 20, distributions, trg, batch_size, max_output_length, gold_strings, predicted_strings, rewards, old_bleus)


    def reinforce(self, max_output_length, src: Tensor, trg: Tensor, src_mask: Tensor,
                        src_length: Tensor, temperature: float) \
        -> Tensor:

        encoder_output, encoder_hidden = self._encode(
            src, src_length,
            src_mask)
        
        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = int(max(src_length.cpu().numpy()) * 1.5)

        batch_size = src_mask.size(0)
        sequence = src_mask.new_full(size=[batch_size, 1], fill_value=self.bos_index,
                                    dtype=torch.long)
        hidden = self.decoder._init_hidden(encoder_hidden)
        attention_vectors = None
        log_probabs = 0
        distributions=[]
        targets=trg.tolist()
        # collect variables for logging
        for i in range(max_output_length):
            previous_words = sequence[:, -1].view(-1, 1)
            logits, hidden, attention_scores, attention_vectors = self.decoder(
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                src_mask=src_mask,
                trg_embed=self.trg_embed(previous_words),
                hidden=hidden,
                prev_att_vector=attention_vectors,
                unroll_steps=1)

            logits = logits.view(-1, logits.size(-1))/temperature
            distrib =  Categorical(logits = logits)
            distributions.append(distrib)
            sampled_word = distrib.sample() 
            log_probabs -= distrib.log_prob(sampled_word)
            sequence = torch.cat([sequence, sampled_word.view(-1, 1)], -1)

        predicted_output = self.trg_vocab.arrays_to_sentences(arrays=sequence,
                                            cut_at_eos=True)
        gold_output = self.trg_vocab.arrays_to_sentences(arrays=trg,
                                            cut_at_eos=True)
        predicted_strings = [join_strings(wordlist) for wordlist in predicted_output]
        gold_strings = [join_strings(wordlist) for wordlist in gold_output]
        batch_loss, rewards, old_bleus = self.loss_function(predicted_strings, gold_strings, log_probabs, top_words, sequence, trg)
        return batch_loss, self.log_peakiness(self.pad_index, self.trg_vocab, 20,distributions, trg, batch_size, max_output_length, gold_strings, predicted_strings, rewards, old_bleus)


    def mrt_transformer(self, max_output_length, src: Tensor, trg: Tensor, src_mask: Tensor,
                        src_length: Tensor, temperature: float, samples: int, alpha: float, add_gold=False) \
        -> Tensor:

        distributions = []
        encoder_output, _ = self._encode(src, src_length,
                    src_mask)
        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = int(max(src_length.cpu().numpy()) * 1.5)
        batch_size = src_mask.size(0)
        #samplthread = threading.Thread(target=self.background_track_calculation, name="pre_calc", args=[state, rgb_code])
        ys = encoder_output.new_full([batch_size, 1], self.bos_index, dtype=torch.long)
        targets = trg.tolist()
        trg_mask = src_mask.new_ones([1, 1, 1])
        total_prob = 0
        collect_gold_probs = 0 
        distributions = []
        # repeat tensor for vectorized solution
        ys = ys.repeat(samples, 1)
        src_mask = src_mask.repeat(samples,1,1)
        encoder_output = encoder_output.repeat(samples,1,1)
        for i in range(max_output_length):
            logits, out, _, _ = self.decoder(
            trg_embed=self.trg_embed(ys),
            encoder_output=encoder_output,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=None,
            hidden=None,
            trg_mask=trg_mask
            )
            logits = logits[:, -1]/temperature
            distrib = Categorical(logits=logits)
            probabilities = distrib.probs
            if i < trg.shape[1]:
                # get ith column 
                # better: take the average 
                ith_column = trg[:,i]
                pumped_ith_column = ith_column.repeat(samples)
                stacked = torch.stack(list(torch.split(distrib.log_prob(pumped_ith_column), batch_size)))
                log_prob = stacked[0]
                # incrementally update gold ranks in every step
                # batch elements
                collect_gold_probs-=log_prob*alpha
            distributions.append(distrib)
        
            next_word = distrib.sample()
            ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
            total_prob -= distrib.log_prob(next_word)*alpha

        all_sequences = list(torch.split(ys, batch_size))
        sentence_probabs= list(torch.split(total_prob, batch_size))
        #distributions.append(sample_distributions)
        ys = ys[:, 1:]
        predicted_outputs = [self.trg_vocab.arrays_to_sentences(arrays=sequ,
                                                        cut_at_eos=True) for sequ in all_sequences]
        gold_output = self.trg_vocab.arrays_to_sentences(arrays=trg,
                                                    cut_at_eos=True)
        predicted_sentences = [[join_strings(wordlist) for wordlist in predicted_output] 
            for predicted_output in predicted_outputs]
        gold_strings = [join_strings(wordlist) for wordlist in gold_output]
        # add gold
        all_gold_sentences = gold_strings*samples
        if add_gold:
            sentence_probabs.append(collect_gold_probs)
            predicted_sentences.append(gold_strings)
            all_gold_sentences.append(gold_strings)
        # calculate Qs
        #sum_of_probabs = sum([probab for probab in sentence_probabs])
        #list_of_Qs = [(probab)/sum_of_probabs for probab in sentence_probabs]
        list_of_Qs = torch.softmax(torch.stack(sentence_probabs), 0)
        # sanity check
        #assert(len(predicted_sentences) == len(list_of_Qs))        
        batch_loss = 0
        for index, Q in enumerate(list_of_Qs):
            for prediction, gold_ref, Q_iter in zip(predicted_sentences[index], all_gold_sentences[index], Q):
                # gradient ascent
                batch_loss  -= bleu([prediction], [gold_ref])*Q_iter
        rewards = [bleu([prediction], [gold_ref]) for prediction, gold_ref in zip(predicted_sentences[-1], all_gold_sentences[-1])]
        Qs_to_return = [q.tolist() for q in list_of_Qs]
        #batch_loss = loss_function(predicted_strings, gold_strings, log_probabs, top_words, sequence, trg)
        return batch_loss, log_peakiness(self.pad_index, self.trg_vocab, 20, distributions, trg, batch_size, max_output_length, gold_strings, predicted_sentences, Qs_to_return, rewards)


    def mrt(self, max_output_length, src: Tensor, trg: Tensor, src_mask: Tensor,
                        src_length: Tensor,  temperature: float, samples: int, alpha: float, add_gold=False) \
                -> Tensor:
        distributions = []
        encoder_output, encoder_hidden = self._encode(
                src, src_length,
                src_mask)
        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = int(max(src_length.cpu().numpy()) * 1.5)
        batch_size = src_mask.size(0)

        # vectorization
        ys = ys.repeat(samples, 1)
        src_mask = src_mask.repeat(samples,1,1)
        encoder_output = encoder_output.repeat(samples,1,1)

        sequence = src_mask.new_full(size=[batch_size, 1], fill_value=self.bos_index,
                                    dtype=torch.long)
        hidden = self.decoder._init_hidden(encoder_hidden)
        attention_vectors = None
        #sample_distributions = []
        top_words = []
        total_prob = 0
        targets = trg.tolist()
        for i in range(max_output_length):
            previous_words = sequence[:, -1].view(-1, 1)
            logits, hidden, attention_scores, attention_vectors = self.decoder(
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                src_mask=src_mask,
                trg_embed=self.trg_embed(previous_words),
                hidden=hidden,
                prev_att_vector=attention_vectors,
                unroll_steps=1)
            logits = logits.view(-1, logits.size(-1))
            #logits = temperature * logits.view(-1, logits.size(-1))
            distrib = Categorical(logits = logits)
            #sample_distributions.append(distrib)
            probabilities=distrib.probs
            sampled_word = distrib.sample() 
            total_prob -= distrib.log_prob(sampled_word)
            sequence = torch.cat([sequence, sampled_word.view(-1, 1)], -1)
            # get probability of gold token 
            if _sample == samples-1:
                top100_probs, top100_probs_probs_index = probabilities.topk(100, largest=True, sorted=True)
                if i < len(trg.tolist()[0]):
                    # get ith column 
                    ith_column = trg[:,i]
                    # incrementally update gold ranks in every step
                    for index, token in enumerate(ith_column.tolist()):
                        if token in top100_probs_probs_index.tolist()[index]:
                            gold_token_ranks[index].append(top100_probs_probs_index.tolist()[index].index(token))
                        else:   
                            gold_token_ranks[index].append(900)
                    # incrementally update gold tokens in every step
                    gold_probability = torch.exp(distrib.log_prob(ith_column))
                    for index in range(len(gold_probability.tolist())):
                        if not ith_column[index] == self.pad_index:
                            gold_probabilities[index].append(gold_probability.tolist()[index])
                    # batch elements

                highest_probs, highest_probs_index = probabilities.topk(10, largest=True, sorted=True)
                highest_words = self.trg_vocab.arrays_to_sentences(arrays=highest_probs_index)
                batch_data = self.calculate_peakiness_and_entropy(distrib)
                for index, data in enumerate(batch_data):
                    if not ith_column[index] == self.pad_index:
                        sentence_probability[index].append(batch_data[index][2])
                        sentence_highest_words[index].append(batch_data[index][1])
                        sentence_highest_word[index].append(batch_data[index][3])
                        sentence_highest_probability[index].append(batch_data[index][4])
                entropy += tensor_to_float(batch_data[0][0])
                top_words.append(highest_words) 

        all_targets.append(trg)
        all_sequences.append(sequence)
        sentence_probabs.append(total_prob)

        predicted_output = self.trg_vocab.arrays_to_sentences(arrays=sequence,
                                            cut_at_eos=True)
        gold_output = self.trg_vocab.arrays_to_sentences(arrays=trg,
                                            cut_at_eos=True)
        predicted_strings = [join_strings(wordlist) for wordlist in predicted_output]
        gold_strings = [join_strings(wordlist) for wordlist in gold_output]
    
        predicted_sentences.append(predicted_strings)
        all_gold_sentences.append(gold_strings)
        all_highest_words.append(top_words)

        # calculate Qs
        if add_gold:
            log_gold_prob=0
            for elem in trg:
                log_gold_prob-= distrib.log_prob(elem)
            sentence_probabs.append(log_gold_prob)
            predicted_sentences.append(gold_strings)
            all_gold_sentences.append(gold_strings)

        sum_of_probabs = sum([probab* alpha for probab in sentence_probabs])
        list_of_Qs = [(probab* alpha)/sum_of_probabs for probab in sentence_probabs]
        # sanity_check
        assert(len(predicted_sentences) == len(list_of_Qs))        
        batch_loss = 0
        for index, Q in enumerate(list_of_Qs):
            for prediction, gold_ref, Q_iter in zip(predicted_sentences[index], all_gold_sentences[index], Q):
                batch_loss  += bleu([prediction], [gold_ref])*Q_iter
        rewards = [bleu([prediction], [gold_ref]) for prediction, gold_ref in zip(predicted_sentences[-1], all_gold_sentences[-1])]
        Qs_to_return = [Q.tolist() for Q in list_of_Qs]
        return batch_loss, log_data(distributions, trg, batch_size, max_output_length, gold_strings, predicted_strings, rewards, old_bleus)


    def a2c_transformer(self, max_output_length, src: Tensor, trg: Tensor, src_mask: Tensor,
                        src_length: Tensor, temperature: float, critic: nn.Module) \
            -> Tensor:
        encoder_output, encoder_hidden = self._encode(
            src, src_length,
            src_mask)
        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = int(max(src_length.cpu().numpy()) * 1.5)
        batch_size = src_mask.size(0)
        ys = encoder_output.new_full([batch_size, 1], self.bos_index, dtype=torch.long)
        trg_mask = src_mask.new_ones([1, 1, 1])
        
        
        entropy = 0 
        targets = trg.tolist()
        output = []
        top_words = []
        log_probs = 0
        distributions = []
        actor_log_probabs = []
        actor_samples = []
        critic_encoder_output, _ = critic._encode(
                src, src_length,
                src_mask)
        critic_logits = []
        critic_sequence = critic_encoder_output.new_full(size=[batch_size, 1], fill_value=self.bos_index,
                                    dtype=torch.long)
        #init dict to track eos
        eos_dict = {i:-1 for i in range(batch_size)}

        finished = src_mask.new_zeros((batch_size)).byte()
        for i in range(max_output_length):
            logits, out, _, _ = self.decoder(
                trg_embed=self.trg_embed(ys),
                encoder_output=encoder_output,
                encoder_hidden=None,
                src_mask=src_mask,
                unroll_steps=None,
                hidden=None,
                trg_mask=trg_mask
            )
            logits = logits[:, -1]/temperature
            distrib = Categorical(logits=logits)
            distributions.append(distrib)
            probabilities = distrib.probs
            
            top100_probs, top100_probs_probs_index = probabilities.topk(100, largest=True, sorted=True)
                    
            if i < len(trg.tolist()[0]):
                # get ith column 
                ith_column = trg[:,i]
                # incrementally update gold ranks in every step
                for index, token in enumerate(ith_column.tolist()):
                    if token in top100_probs_probs_index.tolist()[index]:
                        gold_token_ranks[index].append(top100_probs_probs_index.tolist()[index].index(token))
                    else:   
                        gold_token_ranks[index].append(900)
                    # wrong
                    if not token == self.pad_index:
                        # incrementally update gold tokens in every step
                        gold_probability = torch.exp(distrib.log_prob(ith_column))
                        for index in range(len(gold_probability.tolist())):
                            gold_probabilities[index].append(gold_probability.tolist()[index])
                # batch elements
            highest_probs, highest_probs_index = probabilities.topk(10, largest=True, sorted=True)
            highest_words = self.trg_vocab.arrays_to_sentences(arrays=highest_probs_index)
            batch_data = self.calculate_peakiness_and_entropy(distrib)
            for index, data in enumerate(batch_data):
                if not ith_column[index] == self.pad_index:
                    sentence_probability[index].append(batch_data[index][2])
                    sentence_highest_words[index].append(batch_data[index][1])
                    sentence_highest_word[index].append(batch_data[index][3])
                    sentence_highest_probability[index].append(batch_data[index][4])
            entropy += tensor_to_float(batch_data[0][0])
            top_words.append(highest_words) 
            word = distrib.sample()
            sampled_word = word
            log_probs -= distrib.log_prob(sampled_word)
            actor_samples.append(sampled_word.view(-1, 1))
            ys = torch.cat([ys, sampled_word.unsqueeze(-1)], dim=1)
            actor_log_probabs.append(log_probs)
            sampled_word_list = sampled_word.tolist()
            for index in range(len(sampled_word_list)):
                # 3 is index of eos
                if sampled_word_list[index] == 3:
                    if eos_dict[index] == -1:
                        eos_dict[index] = i
            # unroll critic
            critic_logit, _, _, _ = critic.decoder(
                encoder_output=critic_encoder_output,
                encoder_hidden=None,
                src_mask=src_mask,
                trg_embed=self.trg_embed(sampled_word.view(-1,1)),
                hidden=None,
                prev_att_vector=None,
                unroll_steps=None,
                trg_mask=trg_mask
            )

            critic_logits.append(critic_logit)
            critic_distrib =  Categorical(logits = critic_logit.view(-1, critic_logit.size(-1)))
            critic_sample = critic_distrib.sample()
            critic_sequence = torch.cat([critic_sequence, critic_sample.view(-1, 1)], -1)
            # check if previous symbol was <eos>
            is_eos = torch.eq(sampled_word, self.eos_index)
            finished += is_eos
            # stop predicting if <eos> reached for all elements in batch
            if (finished >= 1).sum() == batch_size:
                break

        ys = ys[:, 1:]
        predicted_output = self.trg_vocab.arrays_to_sentences(arrays=ys,
                                                        cut_at_eos=True)
        gold_output = self.trg_vocab.arrays_to_sentences(arrays=trg,
                                                    cut_at_eos=True)
        predicted_strings = [self.join_strings(wordlist) for wordlist in predicted_output]
        gold_strings = [self.join_strings(wordlist) for wordlist in gold_output]
        bleu_scores = []
        for prediction, gold_ref in zip(predicted_strings, gold_strings):
            bleu_scores.append(bleu([prediction], [gold_ref]))
        critic_logits_tensor = torch.stack(critic_logits)

        # clean critic logits
        for dict_index in eos_dict:
            critic_logits_tensor[eos_dict[dict_index]:,dict_index] = bleu_scores[dict_index]
        critic_logits = torch.unbind(critic_logits_tensor)
        bleu_tensor = torch.FloatTensor(bleu_scores).unsqueeze(1)
        critic_loss = torch.cat([torch.pow(bleu_tensor-logit, 2) for logit in critic_logits]).sum()
        rewards = [(bleu_tensor-logit).squeeze(1) for logit in critic_logits]
        batch_loss = 0
        for log_prob, critic_logit in zip(actor_log_probabs, critic_logits):
            batch_loss += log_prob.unsqueeze(1)*(bleu_tensor-critic_logit)
        batch_loss = batch_loss.sum()
        return [batch_loss, critic_loss], self.log_data(distributions, trg, batch_size, max_output_length, gold_strings, predicted_strings, rewards, old_bleus)

    def a2c(self, max_output_length, src: Tensor, trg: Tensor, src_mask: Tensor,
                    src_length: Tensor, temperature: float, critic: nn.Module) \
        -> Tensor:
        encoder_output, encoder_hidden = self._encode(
            src, src_length,
            src_mask)
        
        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = int(max(src_length.cpu().numpy()) * 1.5)

        batch_size = src_mask.size(0)
        sequence = src_mask.new_full(size=[batch_size, 1], fill_value=self.bos_index,
                                    dtype=torch.long)

        critic_sequence = src_mask.new_full(size=[batch_size, 1], fill_value=self.bos_index,
                                    dtype=torch.long)

        hidden = self.decoder._init_hidden(encoder_hidden)
        attention_vectors = None
        log_probabs_sum = 0.0
        # collect variables for logging
        sentence_probability = [[] for i in range(batch_size)]
        sentence_highest_words = [[] for i in range(batch_size)]
        sentence_highest_word = [[] for i in range(batch_size)]
        sentence_highest_probability = [[] for i in range(batch_size)]
        gold_probabilities = [[] for i in range(batch_size)]
        gold_token_ranks = [[] for i in range(batch_size)]

        entropy = 0 
        targets = trg.tolist()
        distributions = []
        top_words = []
        actor_log_probabs = []
        actor_samples = []
        critic_encoder_output, critic_encoder_hidden = critic._encode(
            src, src_length,
            src_mask)
        critic_logits = []
        critic_attention_vectors = None 
        critic_hidden = critic.decoder._init_hidden(critic_encoder_hidden)
        #init dict to track eos
        eos_dict = {i:-1 for i in range(batch_size)}
        for i in range(max_output_length):
            previous_words = sequence[:, -1].view(-1, 1)
            logits, hidden, attention_scores, attention_vectors = self.decoder(
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                src_mask=src_mask,
                trg_embed=self.trg_embed(previous_words),
                hidden=hidden,
                prev_att_vector=attention_vectors,
                unroll_steps=1)
            logits = logits.view(-1, logits.size(-1))
            distrib =  Categorical(logits = logits)
            probabilities=distrib.probs
            # get probability of gold token 
            top100_probs, top100_probs_probs_index = probabilities.topk(100, largest=True, sorted=True)    
            if i < len(trg.tolist()[0]):
                # get ith column 
                ith_column = trg[:,i]
                # incrementally update gold ranks in every step
                for index, token in enumerate(ith_column.tolist()):
                    if token in top100_probs_probs_index.tolist()[index]:
                        gold_token_ranks[index].append(top100_probs_probs_index.tolist()[index].index(token))
                    else:   
                        gold_token_ranks[index].append(900)
                # incrementally update gold tokens in every step
                gold_probability = torch.exp(distrib.log_prob(ith_column))
                for index in range(len(gold_probability.tolist())):
                    if not ith_column[index] == self.pad_index:
                        gold_probabilities[index].append(gold_probability.tolist()[index])
                # batch elements
            highest_probs, highest_probs_index = probabilities.topk(10, largest=True, sorted=True)
            highest_words = self.trg_vocab.arrays_to_sentences(arrays=highest_probs_index)
            batch_data = self.calculate_peakiness_and_entropy(distrib)
            for index, data in enumerate(batch_data):
                if not ith_column[index] == self.pad_index:
                    sentence_probability[index].append(batch_data[index][2])
                    sentence_highest_words[index].append(batch_data[index][1])
                    sentence_highest_word[index].append(batch_data[index][3])
                    sentence_highest_probability[index].append(batch_data[index][4])
            entropy += tensor_to_float(batch_data[0][0])
            top_words.append(highest_words) 
            sampled_word = distrib.sample() 
            actor_samples.append(sampled_word.view(-1, 1))
            log_probabs = -distrib.log_prob(sampled_word)
            log_probabs_sum -= distrib.log_prob(sampled_word)
            sequence = torch.cat([sequence, sampled_word.view(-1, 1)], -1)
            actor_log_probabs.append(log_probabs)
            sampled_word_list = sampled_word.tolist()
            for index in range(len(sampled_word_list)):
                # 3 is index of eos
                if sampled_word_list[index] == 3:
                    if eos_dict[index] == -1:
                        eos_dict[index] = i
            # unroll critic
            critic_logit, critic_hidden, critic_attention_scores, critic_attention_vectors = critic.decoder(
                encoder_output=critic_encoder_output,
                encoder_hidden=critic_encoder_hidden,
                src_mask=src_mask,
                trg_embed=self.trg_embed(sampled_word.view(-1,1)),
                hidden=critic_hidden,
                prev_att_vector=critic_attention_vectors,
                unroll_steps=1)

            critic_logits.append(critic_logit)
            critic_distrib =  Categorical(logits = critic_logit.view(-1, critic_logit.size(-1)))
            critic_sample = critic_distrib.sample()
            critic_sequence = torch.cat([critic_sequence, critic_sample.view(-1, 1)], -1)
        predicted_output = self.trg_vocab.arrays_to_sentences(arrays=sequence,
                                            cut_at_eos=True)
        gold_output = self.trg_vocab.arrays_to_sentences(arrays=trg,
                                            cut_at_eos=True)
        predicted_strings = [self.join_strings(wordlist) for wordlist in predicted_output]
        gold_strings = [self.join_strings(wordlist) for wordlist in gold_output]
        bleu_scores = []
        for prediction, gold_ref in zip(predicted_strings, gold_strings):
            bleu_scores.append(bleu([prediction], [gold_ref]))
        critic_logits_tensor = torch.stack(critic_logits)

        # clean critic logits
        for dict_index in eos_dict:
            critic_logits_tensor[eos_dict[dict_index]:,dict_index] = bleu_scores[dict_index]
        critic_logits = torch.unbind(critic_logits_tensor)
        bleu_tensor = torch.FloatTensor(bleu_scores).unsqueeze(1).cuda()
        critic_loss = torch.cat([torch.pow(bleu_tensor-logit, 2) for logit in critic_logits]).sum()
        rewards = [(bleu_tensor-logit).squeeze(1) for logit in critic_logits]

        batch_loss = 0
        for log_prob, critic_logit in zip(actor_log_probabs, critic_logits):
            batch_loss += log_prob.unsqueeze(1)*(bleu_tensor-critic_logit)
        batch_loss = batch_loss.sum()
        return [batch_loss, critic_loss], self.log_data(distributions, trg, batch_size, max_output_length, gold_strings, predicted_strings, rewards, old_bleus)




    def forward(self, return_type: str = None, **kwargs) \
            -> (Tensor, Tensor, Tensor, Tensor):
        """ Interface for multi-gpu

        For DataParallel, We need to encapsulate all model call: model.encode(),
        model.decode(), and model.encode_decode() by model.__call__().
        model.__call__() triggers model.forward() together with pre hooks
        and post hooks, which take care of multi-gpu distribution.

        :param return_type: one of {"loss", "encode", "decode"}
        """
        if return_type is None:
            raise ValueError("Please specify return_type: "
                             "{`loss`, `encode`, `decode`}.")

        return_tuple = (None, None, None, None)
        if return_type == "loss":
            assert self.loss_function is not None

            out, _, _, _ = self._encode_decode(
                src=kwargs["src"],
                trg_input=kwargs["trg_input"],
                src_mask=kwargs["src_mask"],
                src_length=kwargs["src_length"],
                trg_mask=kwargs["trg_mask"])

            loss, logging = self.measure_transformer_peakiness(
            src=kwargs["src"],
            trg=kwargs["trg"],
            src_mask=kwargs["src_mask"],
            src_length=kwargs["src_length"],
            max_output_length=kwargs["max_output_length"]
            )
            # compute log probs
            log_probs = F.log_softmax(out, dim=-1)

            # compute batch loss
            batch_loss = self.loss_function(log_probs, kwargs["trg"])

            # return batch loss
            #     = sum over all elements in batch that are not pad
            return_tuple = (batch_loss, logging, None, None)

        elif return_type == "reinforce":
            loss, logging = self.reinforce(
            src=kwargs["src"],
            trg=kwargs["trg"],
            src_mask=kwargs["src_mask"],
            src_length=kwargs["src_length"],
            max_output_length=kwargs["max_output_length"],
            temperature=kwargs["temperature"]
            )
            return_tuple = (loss, logging, None, None)

        elif return_type == "transformer_reinforce":
            loss, logging = self.reinforce_transformer(
            src=kwargs["src"],
            trg=kwargs["trg"],
            src_mask=kwargs["src_mask"],
            src_length=kwargs["src_length"],
            max_output_length=kwargs["max_output_length"],
            temperature=kwargs["temperature"]
            )
            return_tuple = (loss, logging, None, None)

        elif return_type == "mrt":
            loss, logging = self.mrt(
            src=kwargs["src"],
            trg=kwargs["trg"],
            src_mask=kwargs["src_mask"],
            src_length=kwargs["src_length"],
            max_output_length=kwargs["max_output_length"],
            temperature=kwargs["temperature"],
            alpha=kwargs["alpha"],
            samples=kwargs["samples"],
            add_gold=kwargs["add_gold"]
            )
            return_tuple = (loss, logging, None, None)

        elif return_type == "transformer_mrt":
            loss, logging = self.mrt_transformer(
            src=kwargs["src"],
            trg=kwargs["trg"],
            src_mask=kwargs["src_mask"],
            src_length=kwargs["src_length"],
            max_output_length=kwargs["max_output_length"],
            temperature=kwargs["temperature"],
            alpha=kwargs["alpha"],
            samples=kwargs["samples"],
            add_gold=kwargs["add_gold"]
            )
            return_tuple = (loss, logging, None, None)

        elif return_type == "transformer_a2c":
            loss, logging = self.a2c_transformer(
            critic=kwargs["critic"],
            src=kwargs["src"],
            trg=kwargs["trg"],
            src_mask=kwargs["src_mask"],
            src_length=kwargs["src_length"],
            max_output_length=kwargs["max_output_length"],
            temperature=kwargs["temperature"],
            )
            return_tuple = (loss, logging, None, None)

        elif return_type == "a2c":
            loss, logging = self.a2c(
            critic=critic,
            src=kwargs["src"],
            trg=kwargs["trg"],
            src_mask=kwargs["src_mask"],
            src_length=kwargs["src_length"],
            max_output_length=kwargs["max_output_length"],
            temperature=kwargs["temperature"],
            )
            return_tuple = (loss, logging, None, None)

        elif return_type == "encode":
            encoder_output, encoder_hidden = self._encode(
                src=kwargs["src"],
                src_length=kwargs["src_length"],
                src_mask=kwargs["src_mask"])

            # return encoder outputs
            return_tuple = (encoder_output, encoder_hidden, None, None)

        elif return_type == "decode":
            outputs, hidden, att_probs, att_vectors = self._decode(
                trg_input=kwargs["trg_input"],
                encoder_output=kwargs["encoder_output"],
                encoder_hidden=kwargs["encoder_hidden"],
                src_mask=kwargs["src_mask"],
                unroll_steps=kwargs["unroll_steps"],
                decoder_hidden=kwargs["decoder_hidden"],
                att_vector=kwargs.get("att_vector", None),
                trg_mask=kwargs.get("trg_mask", None))

            # return decoder outputs
            return_tuple = (outputs, hidden, att_probs, att_vectors)
        return return_tuple

    # pylint: disable=arguments-differ
    def _encode_decode(self, src: Tensor, trg_input: Tensor, src_mask: Tensor,
                       src_length: Tensor, trg_mask: Tensor = None) \
            -> (Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param src: source input
        :param trg_input: target input
        :param src_mask: source mask
        :param src_length: length of source inputs
        :param trg_mask: target mask
        :return: decoder outputs
        """
        encoder_output, encoder_hidden = self._encode(src=src,
                                                      src_length=src_length,
                                                      src_mask=src_mask)
        unroll_steps = trg_input.size(1)
        return self._decode(encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=src_mask, trg_input=trg_input,
                            unroll_steps=unroll_steps,
                            trg_mask=trg_mask)

    def _encode(self, src: Tensor, src_length: Tensor, src_mask: Tensor) \
            -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param src:
        :param src_length:
        :param src_mask:
        :return: encoder outputs (output, hidden_concat)
        """
        return self.encoder(self.src_embed(src), src_length, src_mask)

    def _decode(self, encoder_output: Tensor, encoder_hidden: Tensor,
                src_mask: Tensor, trg_input: Tensor,
                unroll_steps: int, decoder_hidden: Tensor = None,
                att_vector: Tensor = None, trg_mask: Tensor = None) \
            -> (Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unrol the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param att_vector: previous attention vector (optional)
        :param trg_mask: mask for target steps
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        return self.decoder(trg_embed=self.trg_embed(trg_input),
                            encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=src_mask,
                            unroll_steps=unroll_steps,
                            hidden=decoder_hidden,
                            prev_att_vector=att_vector,
                            trg_mask=trg_mask)

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return "%s(\n" \
               "\tencoder=%s,\n" \
               "\tdecoder=%s,\n" \
               "\tsrc_embed=%s,\n" \
               "\ttrg_embed=%s)" % (self.__class__.__name__, self.encoder,
                                    self.decoder, self.src_embed,
                                    self.trg_embed)


class _DataParallel(nn.DataParallel):
    """ DataParallel wrapper to pass through the model attributes """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def build_model(cfg: dict = None,
                src_vocab: Vocabulary = None,
                trg_vocab: Vocabulary = None,
                is_critic: bool = False) -> Model:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param src_vocab: source vocabulary
    :param trg_vocab: target vocabulary
    :return: built and initialized model
    """
    src_padding_idx = src_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = trg_vocab.stoi[PAD_TOKEN]

    src_embed = Embeddings(
        **cfg["encoder"]["embeddings"], vocab_size=len(src_vocab),
        padding_idx=src_padding_idx)

    # this ties source and target embeddings
    # for softmax layer tying, see further below
    if cfg.get("tied_embeddings", False):
        if src_vocab.itos == trg_vocab.itos:
            # share embeddings for src and trg
            trg_embed = src_embed
        else:
            raise ConfigurationError(
                "Embedding cannot be tied since vocabularies differ.")
    else:
        trg_embed = Embeddings(
            **cfg["decoder"]["embeddings"], vocab_size=len(trg_vocab),
            padding_idx=trg_padding_idx)

    # build encoder
    enc_dropout = cfg["encoder"].get("dropout", 0.)
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    if cfg["encoder"].get("type", "recurrent") == "transformer":
        assert cfg["encoder"]["embeddings"]["embedding_dim"] == \
            cfg["encoder"]["hidden_size"], \
            "for transformer, emb_size must be hidden_size"

        encoder = TransformerEncoder(**cfg["encoder"],
                                    emb_size=src_embed.embedding_dim,
                                    emb_dropout=enc_emb_dropout)
    else:
        encoder = RecurrentEncoder(**cfg["encoder"],
                                    emb_size=src_embed.embedding_dim,
                                    emb_dropout=enc_emb_dropout)

    # build decoder
    dec_dropout = cfg["decoder"].get("dropout", 0.)
    dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
    if cfg["decoder"].get("type", "recurrent") == "transformer":
        if is_critic: 
            decoder = CriticTransformerDecoder(
            **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)
        else:
            decoder = TransformerDecoder(
                **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
                emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)
    else:
        if is_critic: 
            decoder = CriticDecoder(
            **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)
        else:
            decoder = RecurrentDecoder(
                **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
                emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)

    model = Model(encoder=encoder, decoder=decoder,
                  src_embed=src_embed, trg_embed=trg_embed,
                  src_vocab=src_vocab, trg_vocab=trg_vocab)
    if not is_critic:
        # tie softmax layer with trg embeddings
        if cfg.get("tied_softmax", False):
            if trg_embed.lut.weight.shape == \
                    model.decoder.output_layer.weight.shape:
                # (also) share trg embeddings and softmax layer:
                model.decoder.output_layer.weight = trg_embed.lut.weight
            else:
                raise ConfigurationError(
                    "For tied_softmax, the decoder embedding_dim and decoder "
                    "hidden_size must be the same."
                    "The decoder must be a Transformer.")
            
    # custom initialization of model parameters
    initialize_model(model, cfg, src_padding_idx, trg_padding_idx)

    return model

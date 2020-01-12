# Load the model in fairseq
import torch
from fairseq.models.bart import BARTModel
from fairseq.data.data_utils import collate_tokens
from transformers import AutoTokenizer

def run_model(model, tokens, return_all_hiddens=False, features_only=False):
    src_lengths = None
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
        src_lengths = torch.Tensor([tokens.shape[1]]).long()
    if tokens.size(-1) > min(model.max_positions()):
        raise ValueError('tokens exceeds maximum length: {} > {}'.format(
            tokens.size(-1), model.max_positions()
        ))
    prev_output_tokens = tokens.clone()
    prev_output_tokens[:, 0] = tokens[:, -1]
    prev_output_tokens[:, 1:] = tokens[:, :-1]
    features, extra = model(
        src_tokens=tokens,
        src_lengths=src_lengths,
        prev_output_tokens=prev_output_tokens,
        features_only=features_only,
        return_all_hiddens=return_all_hiddens,
    )
    if return_all_hiddens:
        # convert from T x B x C -> B x T x C
        inner_states = extra['inner_states']
        return [inner_state.transpose(0, 1) for inner_state in inner_states]
    else:
        return features  # just the last layer's features


bart = BARTModel.from_pretrained('/export/share/rmeng/tools/torchhub/bart.large', checkpoint_file='model.pt')
bart_cnndm = BARTModel.from_pretrained('/export/share/rmeng/tools/torchhub/bart.large.cnn', checkpoint_file='model.pt')
gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir='/export/share/rmeng/output/pretrain_cache/')
roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base', cache_dir='/export/share/rmeng/output/pretrain_cache/')

bart.eval()  # disable dropout (or leave in train mode to finetune)
bart_cnndm.eval()  # disable dropout (or leave in train mode to finetune)

test_str = 'Helloworld!'
bart_tokens = bart.encode(test_str) # BART will add <s> and </s>
bart_cnndm_tokens = bart_cnndm.encode(test_str)
roberta_tokens = roberta_tokenizer.encode(test_str)

# assert tokens.tolist() == [0, 31414, 232, 328, 2]
print(bart_tokens)
print(bart_cnndm_tokens)
print(roberta_tokens)

# Extract the last layer's features
output = run_model(bart.model, bart_tokens)
last_layer_features = bart.extract_features(bart_tokens)
assert last_layer_features.size() == torch.Size([1, 5, 1024])

# Extract all layer's features from decoder (layer 0 is the embedding layer)
all_layers = bart.extract_features(bart_tokens, return_all_hiddens=True)
assert len(all_layers) == 13
assert torch.all(all_layers[-1] == last_layer_features)

# Download BART already finetuned for MNLI
bart = torch.hub.load('pytorch/fairseq', 'bart.large.mnli')
bart.eval()  # disable dropout for evaluation

# Encode a pair of sentences and make a prediction
tokens = bart.encode('BART is a seq2seq model.', 'BART is not sequence to sequence.')
bart.predict('mnli', tokens).argmax()  # 0: contradiction

# Encode another pair of sentences
tokens = bart.encode('BART is denoising autoencoder.', 'BART is version of autoencoder.')
bart.predict('mnli', tokens).argmax()  # 2: entailment


bart.register_classification_head('new_task', num_classes=3)
logprobs = bart.predict('new_task', tokens)


bart = torch.hub.load('pytorch/fairseq', 'bart.large.mnli')
bart.eval()

batch_of_pairs = [
    ['BART is a seq2seq model.', 'BART is not sequence to sequence.'],
    ['BART is denoising autoencoder.', 'BART is version of autoencoder.'],
]

batch = collate_tokens(
    [bart.encode(pair[0], pair[1]) for pair in batch_of_pairs], pad_idx=1
)

logprobs = bart.predict('mnli', batch)
print(logprobs.argmax(dim=1))
# tensor([0, 2])


bart.cuda()
bart.predict('new_task', tokens)



label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
ncorrect, nsamples = 0, 0
bart.cuda()
bart.eval()
with open('glue_data/MNLI/dev_matched.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[8], tokens[9], tokens[-1]
        tokens = bart.encode(sent1, sent2)
        prediction = bart.predict('mnli', tokens).argmax().item()
        prediction_label = label_map[prediction]
        ncorrect += int(prediction_label == target)
        nsamples += 1
        print('| Accuracy: ', float(ncorrect)/float(nsamples))
# Expected output: 0.9010
import torch
import torchaudio
from Gemini_df_resnet import Gemini_df_resnet179
import torchaudio.compliance.kaldi as kaldi
import torch.nn.functional as F

def cosine_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two embeddings.

    Args:
        embedding1 (torch.Tensor): First embedding of shape (1, 256).
        embedding2 (torch.Tensor): Second embedding of shape (1, 256).

    Returns:
        float: Cosine similarity between the two embeddings.
    """
    assert embedding1.shape == (1, 256), "embedding1 must have shape (1, 256)"
    assert embedding2.shape == (1, 256), "embedding2 must have shape (1, 256)"

    # Normalize the embeddings to unit vectors
    embedding1_normalized = F.normalize(embedding1, p=2, dim=1)
    embedding2_normalized = F.normalize(embedding2, p=2, dim=1)

    # Compute cosine similarity
    similarity = torch.sum(embedding1_normalized * embedding2_normalized)

    return similarity.item()

def compute_fbank(waveform,
                  sample_rate,
                  num_mel_bins=80,
                  frame_length=25,
                  frame_shift=10,
                  dither=1.0,
                  flag_apply_cmvn=True):
    """ Extract fbank
    """
    waveform = waveform * (1 << 15)
    mat = kaldi.fbank(waveform,
                      num_mel_bins=num_mel_bins,
                      frame_length=frame_length,
                      frame_shift=frame_shift,
                      dither=dither,
                      sample_frequency=sample_rate,
                      window_type='hamming',
                      use_energy=False)
    if flag_apply_cmvn:
        mat = apply_cmvn(mat=mat)
        print("Notification: cmvn is applied")
    return mat

def apply_cmvn(mat, norm_mean=True, norm_var=False):
    """ Apply CMVN
    """
    if norm_mean:
        mat = mat - torch.mean(mat, dim=0)
    if norm_var:
        mat = mat / torch.sqrt(torch.var(mat, dim=0) + 1e-8)
    return mat

# load model
sv_net = Gemini_df_resnet179(80, 256)
sv_net.load_state_dict(torch.load('./0621-Gemini_df_resnet179-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-AdamW-epoch165-LM/models/final_model.pt'), strict=False)
sv_net.to('cuda')

# example wav paths
wav_path_spk1_utt1 = '/home/tianchi/data/VoxCeleb/voxceleb1/wav/id10050/6OUWWa4tdJw/00001.wav'
wav_path_spk1_utt2 = '/home/tianchi/data/VoxCeleb/voxceleb1/wav/id10050/Yo0U6EbyVJg/00001.wav'
wav_path_spk2_utt1 = '/home/tianchi/data/VoxCeleb/voxceleb1/wav/id10051/hEnGRr7qNUY/00001.wav'
wav_path_spk2_utt2 = '/home/tianchi/data/VoxCeleb/voxceleb1/wav/id10051/v8znF6-r-D8/00001.wav'

# load wav files
wav_spk1_utt1, sample_rate_spk1_utt1 = torchaudio.load(wav_path_spk1_utt1)
wav_spk1_utt2, sample_rate_spk1_utt2 = torchaudio.load(wav_path_spk1_utt2)
wav_spk2_utt1, sample_rate_spk2_utt1 = torchaudio.load(wav_path_spk2_utt1)
wav_spk2_utt2, sample_rate_spk2_utt2 = torchaudio.load(wav_path_spk2_utt2)
assert sample_rate_spk1_utt1 == sample_rate_spk1_utt2 == sample_rate_spk2_utt1 == sample_rate_spk2_utt2 == 16000

# extract Fbank and apply cmvn
fea_spk1_utt1 = compute_fbank(wav_spk1_utt1, sample_rate_spk1_utt1, flag_apply_cmvn=True)
fea_spk1_utt2 = compute_fbank(wav_spk1_utt2, sample_rate_spk1_utt2, flag_apply_cmvn=True)
fea_spk2_utt1 = compute_fbank(wav_spk2_utt1, sample_rate_spk2_utt1, flag_apply_cmvn=True)
fea_spk2_utt2 = compute_fbank(wav_spk2_utt2, sample_rate_spk2_utt2, flag_apply_cmvn=True)

# extract embeddings
embd_spk1_utt1 = sv_net(fea_spk1_utt1.to('cuda').unsqueeze(0))
embd_spk1_utt2 = sv_net(fea_spk1_utt2.to('cuda').unsqueeze(0))
embd_spk2_utt1 = sv_net(fea_spk2_utt1.to('cuda').unsqueeze(0))
embd_spk2_utt2 = sv_net(fea_spk2_utt2.to('cuda').unsqueeze(0))

# output cosine similarity
print('\n--- inference for same speaker, the cosine similarity should be closer to 1.0 ---')
print("cosine similarity of spk1_utt1 and spk1_utt2:", cosine_similarity(embd_spk1_utt1, embd_spk1_utt2))
print("cosine similarity of spk2_utt1 and spk2_utt2:", cosine_similarity(embd_spk2_utt1, embd_spk2_utt2))
print('\n--- inference for different speaker, the cosine similarity should be closer to 0.0 ---')
print("cosine similarity of spk1_utt1 and spk2_utt1:", cosine_similarity(embd_spk1_utt1, embd_spk2_utt1))
print("cosine similarity of spk1_utt1 and spk2_utt2:", cosine_similarity(embd_spk1_utt1, embd_spk2_utt2))
print("cosine similarity of spk1_utt2 and spk2_utt1:", cosine_similarity(embd_spk1_utt2, embd_spk2_utt1))
print("cosine similarity of spk1_utt2 and spk2_utt2:", cosine_similarity(embd_spk1_utt2, embd_spk2_utt2))

# outputs of the demo
'''
Notification: cmvn is applied
Notification: cmvn is applied
Notification: cmvn is applied
Notification: cmvn is applied

--- inference for same speaker, the cosine similarity should be closer to 1.0 ---
cosine similarity of spk1_utt1 and spk1_utt2: 0.6580173373222351
cosine similarity of spk2_utt1 and spk2_utt2: 0.6858957409858704

--- inference for different speaker, the cosine similarity should be closer to 0.0 ---
cosine similarity of spk1_utt1 and spk2_utt1: 0.14298376441001892
cosine similarity of spk1_utt1 and spk2_utt2: 0.1417258083820343
cosine similarity of spk1_utt2 and spk2_utt1: 0.06481592357158661
cosine similarity of spk1_utt2 and spk2_utt2: 0.09413496404886246
'''

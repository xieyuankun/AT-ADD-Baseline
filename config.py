import argparse

def initParams():
    parser = argparse.ArgumentParser(description="Configuration for the project")

    parser.add_argument('--seed', type=int, help="Random number seed for reproducibility", default=688)

    # Train & Dev Data folder prepare
    parser.add_argument("--atadd_t1_train_audio", type=str, help="Path to the training audio for ATADD T1 dataset",
                        default='yourpath/atadd/T1/train')
    parser.add_argument("--atadd_t1_train_label", type=str, help="Path to the training label for ATADD T1 dataset",
                        default="yourpath/atadd/T1/label/train.csv")
    parser.add_argument("--atadd_t1_dev_audio", type=str, help="Path to the development audio for ATADD T1 dataset",
                        default='yourpath/atadd/T1/dev')
    parser.add_argument("--atadd_t1_dev_label", type=str, help="Path to the development label for ATADD T1 dataset",
                        default="yourpath/atadd/T1/label/dev.csv")
    parser.add_argument("--atadd_t1_eval_audio", type=str, help="Path to the evaluation audio for ATADD T1 dataset",
                        default='yourpath/atadd/T1/eval')
    parser.add_argument("--atadd_t1_eval_label", type=str, help="Path to the evaluation label for ATADD T1 dataset",
                        default="yourpath/atadd/T1/label/eval.csv")

    parser.add_argument("--atadd_t2_train_audio", type=str, help="Path to the training audio for ATADD T2 dataset",
                        default='yourpath/atadd/T2/train')
    parser.add_argument("--atadd_t2_train_label", type=str, help="Path to the training label for ATADD T2 dataset",
                        default="yourpath/atadd/T2/label/train.csv")
    parser.add_argument("--atadd_t2_dev_audio", type=str, help="Path to the development audio for ATADD T2 dataset",
                        default='yourpath/atadd/T2/dev')
    parser.add_argument("--atadd_t2_dev_label", type=str, help="Path to the development label for ATADD T2 dataset",
                        default="yourpath/atadd/T2/label/dev.csv")
    parser.add_argument("--atadd_t2_eval_audio", type=str, help="Path to the evaluation audio for ATADD T2 dataset",
                        default='yourpath/atadd/T2/eval')
    parser.add_argument("--atadd_t2_eval_label", type=str, help="Path to the evaluation label for ATADD T2 dataset",
                        default="yourpath/atadd/T2/label/eval.csv")

    # SSL folder prepare
    parser.add_argument("--xlsr", default="yourpath/huggingface/wav2vec2-xls-r-300m")
    parser.add_argument("--wavlm", default="yourpath/huggingface/wavlm-large/")
    parser.add_argument("--mert", default="yourpath/huggingface/MERT-300M/")

    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=False, default='./models/try/')

    # countermeasure
    parser.add_argument("--audio_len", type=int, help="raw waveform length", default=64600)
    parser.add_argument('-m', '--model', help='Model arch', default='pt-w2v2aasist',
                        choices=['specresnet', 'aasist', 'ft-w2v2aasist', 'fr-wavlmaasist', 'fr-mertaasist',
                                 'fr-w2v2aasist', 'ft-wavlmaasist', 'ft-mertaasist',
                                 'pt-w2v2aasist', 'wpt-w2v2aasist',
                                 'pt-wavlmaasist', 'wpt-wavlmaasist',
                                 'pt-mertaasist', 'wpt-mertaasist'])

    # pt
    parser.add_argument("--prompt_dim", type=int, help="prompt dim", default=1024)
    parser.add_argument("--num_prompt_tokens", type=int, help="audio dim", default=10)
    parser.add_argument("--pt_dropout", type=float, help="dropout", default=0.1)

    # wpt
    parser.add_argument("--num_wavelet_tokens", type=int, help="wavelet token", default=4)

    return parser

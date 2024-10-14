import sentencepiece as spm

# SMILES字符串
smiles = "N(C)C(=O)Cc1n2cc(C)ccc2nc1c3ccc(C)cc3"

# 训练BPE模型
vocab_size = 16  # 词汇表大小
character_coverage = 0.99  # 设置字符覆盖率为0.99

# 创建训练数据
train_data = [smiles] * 10000  # 复制字符串，以便有足够的数据量训练模型
with open('train_smiles.txt', 'w') as f:
    for s in train_data:
        f.write(s + '\n')

# 训练模型
spm.SentencePieceTrainer.train(
    '--input=train_smiles.txt '
    '--model_prefix=bpe_model '
    '--vocab_size={} '
    '--character_coverage={}'.format(vocab_size, character_coverage)
)

# 加载模型
sp = spm.SentencePieceProcessor()
sp.Load('bpe_model.model')

# 对SMILES字符串进行切分
encoded_smiles = sp.EncodeAsPieces(smiles)
print(encoded_smiles)
import os

from word_gan.settings import SETTINGS

MODEL_PATH = os.path.join(SETTINGS.DATA_DIR, 'model.txt')

TOKENS_PATH = os.path.join(SETTINGS.VOCAB_PATH, 'tokens.txt')

stopwords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
             'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into',
             'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the',
             'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were',
             'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to',
             'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have',
             'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can',
             'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
             'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by',
             'doing', 'it', 'how', 'further', 'was', 'here', 'than'}


def extract_tokens_from_model(path, out_path):
    with open(out_path, 'w') as out:
        # out.write('@@PADDING@@\n')
        out.write('@@UNKNOWN@@\n')
        out.write('EOS\n')
        out.write('BOS\n')

        for word in stopwords:
            out.write(f'{word}\n')

        with open(path) as fd:
            next(fd)
            for line in fd:
                word, _ = line.split(' ', 1)
                if word not in stopwords:
                    out.write(word)
                    out.write('\n')


if __name__ == '__main__':
    extract_tokens_from_model(MODEL_PATH, TOKENS_PATH)

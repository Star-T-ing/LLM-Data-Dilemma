import os
import jieba
import numpy as np
from tqdm import tqdm

from gensim.models import KeyedVectors, TfidfModel
from gensim.corpora import Dictionary
from gensim import matutils
from itertools import islice

wv_path = '/home/ruansikai/Limerence/assignments/LLM/src/tfidf_word2vec/tecent-200d/light_Tencent_AILab_ChineseEmbedding.bin'

def isChinese(word):
    """是否为中文字符
    :param word:
    :return:
    """
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

class EmbedReplace():
    def __init__(self, samples):
        self.samples = samples
        self.samples = [list(jieba.cut(sample)) for sample in self.samples]
        self.wv = KeyedVectors.load_word2vec_format(wv_path, binary=True)

        if os.path.exists('/home/ruansikai/Limerence/assignments/LLM/src/tfidf_word2vec/tfidf.model'):
            self.tfidf_model = TfidfModel.load('/home/ruansikai/Limerence/assignments/LLM/src/tfidf_word2vec/tfidf.model')
            self.dct = Dictionary.load('/home/ruansikai/Limerence/assignments/LLM/src/tfidf_word2vec/tfidf.dict')
            self.corpus = [self.dct.doc2bow(doc) for doc in self.samples]
        else:
            self.dct = Dictionary(self.samples)
            self.corpus = [self.dct.doc2bow(doc) for doc in self.samples]
            self.tfidf_model = TfidfModel(self.corpus)
            self.dct.save('/home/ruansikai/Limerence/assignments/LLM/src/tfidf_word2vec/tfidf.dict')
            self.tfidf_model.save('/home/ruansikai/Limerence/assignments/LLM/src/tfidf_word2vec/tfidf.model')
            self.vocab_size = len(self.dct.token2id)

    def vectorize(self, docs, vocab_size):
        return matutils.corpus2dense(docs, vocab_size)

    def extract_keywords(self, dct, tfidf, threshold=0.2, topk=5):
        """ 提取关键词
        :param dct (Dictionary): gensim.corpora.Dictionary
        :param tfidf (list):
        :param threshold: tfidf的临界值
        :param topk: 前 topk 个关键词
        :return: 返回的关键词列表
        """
        tfidf = sorted(tfidf, key=lambda x: x[1], reverse=True)

        return list(islice([dct[w] for w, score in tfidf if score > threshold], topk))

    def replace(self, sample, doc):
        """用wordvector的近义词来替换，并避开关键词

        :param sample (list): reference token list
        :param doc (list): A reference represented by a word bag model
        :return: 新的文本
        """
        keywords = self.extract_keywords(self.dct, self.tfidf_model[doc])
        #
        num = int(len(sample) * 0.3)
        new_tokens = sample.copy()
        indexes = np.random.choice(len(sample), num)
        for index in indexes:
            token = sample[index]
            if isChinese(token) and token not in keywords and token in self.wv:
                new_tokens[index] = self.wv.most_similar(positive=token, negative=None, topn=1)[0][0]

        return ''.join(new_tokens)

    def generate_samples(self):
        """
        得到用word2vector词表增强后的数据
        :param write_path:
        """
        replaced = []
        pbar = tqdm(len(self.samples))
        for sample, doc in zip(self.samples, self.corpus):
            replaced.append(self.replace(sample, doc))
            pbar.update(1)
        
        return replaced
    
def replace(sample_list):
    replacer = EmbedReplace(sample_list)
    return replacer.generate_samples()

if __name__ == '__main__':
    samples = [
        '高考冲刺 校门口送饭家长排成行(图) “在学习上我不能给予帮助，只能从生活上关心她！”离高考不足一个月，在广大考生全力冲刺准备迎考时，很多家长也变着花样每天做出营养餐送到学校给孩子开“小灶”补充营养。 昨天上午11点40分，市一中校门外，离下课还有半小时，等待送饭的家长在校门口站成了一排，手里拎着各式各样的饭盒。等到校门一打开，家长们就涌进学校的花园里，为刚下课的孩子送上可口的饭菜。 家住江北南桥寺的刘大爷给外孙女婷婷(化名)准备了糖醋排骨、虾、鸡汤。刘大爷说，自从外孙女上高三以来，学习压力很大，每天早上6点起床，晚上10点多才回家，他每天中午都要往返于江北和沙坪坝，给婷婷送饭到学校，风雨无阻。希望给婷婷增加营养，同时，让她放宽心，好好备考。 花园的另一角，家住渝北的李女士给女儿小雅送来了老鸭汤。李女士说，自己家在江北，但为了方便女儿学习，从高三开始，一家人就在学校附近租了一套房子。利用每周两天的休息时间，她会自己准备午餐送到学校。小雅称，妈妈每次送来自己最喜欢的饭菜，让自己在紧张学习的同时，感到很轻松。据了解，临考前，各中学专为孩子送饭的家长不少。 一中办公室主任张群力表示，一直以来，学校食堂都为学生准备比较清淡、营养较好的饭菜，而每年临近高考、中考，食堂会对饮食进行科学搭配，确保学生营养。对于家长送饭到学校，张群力对此表示理解和支持，他认为，家长对孩子饮食上的关怀，跟孩子是一种心理上的交流，有助于减轻孩子学习上的负担。 八中总务处副主任周明介绍，学校食堂专门配备了兼职营养师，给同学们提供的都是营养套餐，家长们不必担心。他认为，临考前，许多家长为孩子送饭到学校，不仅有益于孩子的身体，对学生心理也是一种安慰，八中每天到校为孩子送餐的家长也很多。 重庆铁路中学校长黄兴力对此却不很主张，他认为，为让学生轻松迎考，学校食堂的饭菜都经过精心的营养合理搭配后推出的，且很注重各方面的后勤服务，虽然家长每天送饭是一种关心，但对于孩子来说，过分关注反倒会增加心理压力。 饮食支招 考前饮食不要大变 每天宜吃两个水果 中高考在即，如何合理科学的为考生们安排饮食，既是学校也是很多家长关心的问题，昨日记者对此采访了第三军医大学教授、市营养学会专家石元刚为广大考生及家长支招。 石元刚表示，考生考试前饮食不要因中高考临近而刻意改变，临考前一段时间及考试期间，饮食量都不要比平时增加太多。他称，主食不是可有可无，考生的饮食要保证主食的摄入量，鱼类、肉类只能补充人体所需的蛋白质，而大脑的思维主要依靠的是葡萄糖，只有主食才能转化为葡萄糖，这就需要每天摄取一定量的主食。 水果蔬菜含有丰富的营养素及各种维生素和矿物质，还有缓解厌食及便秘的作用。考生应保证每天吃两个水果，另外，粗纤维的蔬菜要少吃，如果平时没有常吃的习惯考前一定不要突然增加。而菠菜、胡萝卜可增强记忆力，洋葱能改善大脑供血，帮助考生集中精神，这类食物可适当增加一些。 石教授表示，考生如果考前压力大，产生厌食感，家长可以把每日三餐变成每日四五餐，增加进餐次数，在控制总量的前提下，多餐分吃，考生也同样可以摄取到一天所需的营养量。',
        '尼康遭Intellectual Ventures侵权起诉 彭博社消息，专利公司Intellectual Ventures(以下简称IV)近日对尼康提出4项数码相机技术的专利侵权诉讼。 在向法院递交的文件中，IV称他们早在2008年就接洽过尼康，讨论有关专利授权的问题。但尼康当时拒绝会面。2011年，他们再次与尼康接触，而他们“极有诚意达成共识的努力”也最终失败。 IV此次对尼康提起的起诉涉及4项技术专利： * 专利6,121,960(2000年9月注册)——触摸屏系统及方法 * 专利6,181,836(2001年1月注册)——无损图片编辑系统及方法 * 专利6,221,686(2001年4月注册)——半导体图像传感器制造方法 * 专利6,979,587(2005年12月注册)——图像传感器及制造方法 根据法庭文件，IV目前宣布拥有“超过35000项”专利资产。他们称已向独立发明者支付超过4亿美元购买发明，而通过专利授权已获利20亿美元。 Intellectual Ventures成立于2000年，由前微软首席技术官纳森·梅尔沃德(Nathan Myhrvold)创立。根据彭博社数据，自2010年12月以来，IV已经发起过至少7宗专利侵权案。',
        '七款果蔬色拉美味又低卡 夏天吃出好身材 导读：夏季食欲减退，又是减肥最迫切的时候，健康的蔬果沙拉既美味又能美肤瘦身，是上等选择。新鲜水果还能解除体内堆积的毒素和废物，把积存在细胞中的毒素溶解，并排出体外。下面就介绍7款美味的蔬果沙拉，不但美味，减肥美肤的效果也没话说！ 1、四色果香沙拉 材料：樱桃、葡萄、荔枝、圣女果各5颗，柠檬汁、沙拉酱适量。 做法：将樱桃、葡萄、圣女果分别清洗干净，荔枝剥皮去核，用沙拉酱调拌均匀，最后加入柠檬汁。 功效：樱桃有很高的药用价值，能够温肠通便；葡萄可以帮助肝脏排毒；荔枝不但可以补肾，还可以加速肾脏的废物代谢；柠檬含有的高度碱性可改善血液循环，排除肺部毒素。 2、草莓鲜果沙拉 材料：草莓若干，苹果一个，香蕉一根，酸奶一杯，蜂蜜适量。 做法：将草莓、苹果洗净切块，香蕉切成小段，与酸奶混合，加入适量蜂蜜。 功效：草莓和苹果里面都含有大量果胶，可以清洁肠胃；香蕉可润肠通便，对治疗便秘有辅助作用；蜂蜜和酸奶更是排毒养颜的佳品。 3、缤纷蔬菜沙拉 材料：洋葱、黄瓜、胡萝卜、黑木耳适量，橄榄油、黑胡椒、盐少许。 做法：将洋葱、黄瓜切丁，将胡萝卜和黑木耳用水焯熟后切成小片，倒入橄榄油，将黄瓜丁、洋葱丁、胡萝卜和黑木耳调拌均匀，最后加入黑胡椒和盐调味。 功效：黄瓜的葫芦素和黄瓜酸可以帮助肝脏排毒，并且有利尿的功效，可以排除肾脏的代谢物；黑木耳可以吸附肠道内的杂质，净化血液；胡萝卜可以降低血液中的汞浓度。 4、蜜桃西柚沙拉 材料：西柚半个，梨、桃各一个，沙拉酱，蜂蜜适量。 做法：将梨、桃分别洗净，柚子去皮，将三者切块，倒入沙拉酱搅匀，最后加入蜂蜜。 功效：桃子具有生津润肠、活血消积的功效；梨中类黄酮物质和抗氧化营养成分能够排除肠道中的致癌物质，防止人体排毒系统受到伤害；柚子则可以健胃去火，润肠通便。',
        
    ]
    replacer = EmbedReplace(samples)
    print(replacer.generate_samples())

    # file = ''
    # wv_from_text = KeyedVectors.load_word2vec_format(file, binary=True)
    # wv_from_text.init_sims(replace=True)  # 神奇，很省内存，可以运算most_similar
    # print(wv_from_text.most_similar("广阔"))
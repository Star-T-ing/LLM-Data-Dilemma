import os
import re

def extract(sample: str):
    pos = re.search(r"[，。！](?! )", sample)
    
    if pos is None:
        return None, None
    start = pos.start()
    idx = sample.rfind(' ', 0, start)
    if idx == -1:
        return None, None
    return sample[:idx], sample[idx + 1:]
        
if __name__ == "__main__":
    sample = "让你工作更舒心 主流精英商务本导购 作者：阳仔 现在笔记本的价格越来越便宜，越来越多的普通家庭用户也拥有了属于自己的笔记本电脑。但是在众多低价的笔记本当中，我们很难发现有商务笔记本的踪影。造成这种局面的原因是商业用户和家庭用户对电脑的要求有很大的不同，对于商业用户来说，最重要的不是电脑的性能，多媒体能力和外观，而是系统的稳定性，安全性，售后服务和技术支持的能力。由于商务笔记本的门槛比较高，所以在质量上各家都控制的很严格，此次笔者就给大家挑选了几款现在比较受关注的商务笔记本，供大家参考。 今日登陆中关村的ThinkPad T400s系列机型共有2款，详细型号和参数请大家参阅下表： 作为一款革命性的产品，ThinkPad T400s的变化主要体现在两个方面，在维持了ThinkPad笔记本传统的元素之外，机器的外观设计相比T400系列而言有着非常大的改变，机器除了变得更加轻薄之外，接口的设计产生了许多变化，除了左右两侧接口位置改变之外，机器的后部，也容纳了数个接口。 另外，T400s的操作界面也有着不小的改变：Escape和 Delete键就要比其他机子上的按键大，同时将麦克风和扬声器的静音键分了开来。这些按键均拥有自己的LED指示灯。同时联想还对Caps Lock键、电源的LED指示灯进行了改进，并且增强了指纹识。"
    print(extract(sample))
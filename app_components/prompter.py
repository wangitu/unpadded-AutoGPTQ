"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import List, Union, Optional


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", prompt_bos=None, prompt_eos=None, system=None, verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join(osp.dirname(osp.abspath(__file__)), "templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
            
        if prompt_bos is not None:
            self.template["prompt_bos"] = prompt_bos
        self.template["prompt_input"] = self.template["prompt_input"].replace("<bos>", self.prompt_bos)
        self.template["prompt_no_input"] = self.template["prompt_no_input"].replace("<bos>", self.prompt_bos)
        self.template["prompt_history"] = self.template["prompt_history"].replace("<bos>", self.prompt_bos)
        
        if prompt_eos is not None:
            self.template["prompt_eos"] = prompt_eos
        self.template["prompt_history"] = self.template["prompt_history"].replace("<eos>", self.prompt_eos)
        
        if system is not None:
            self.template["system"] = system
        
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: Union[str, List],
        input: Optional[str] = None,
        label: Union[str, List] = None,
        user_prefix: Optional[str] = None,
        gpt_prefix: Optional[str] = None
    ) -> str:
        if isinstance(instruction, list):
            return self.chat_prompt(instruction, label, user_prefix=user_prefix, gpt_prefix=gpt_prefix)
        
        user_prefix = user_prefix or self.template["user_prefix"]
        gpt_prefix = gpt_prefix or self.template["gpt_prefix"]
        
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input, user_prefix=user_prefix, gpt_prefix=gpt_prefix, system=self.system
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction, user_prefix=user_prefix, gpt_prefix=gpt_prefix, system=self.system
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res
    
    def chat_prompt(
        self, instructions: List, labels: List, 
        user_prefix: Optional[str] = None, gpt_prefix: Optional[str] = None,
        user_prompt: bool = False
    ):
        assert isinstance(instructions, list) and instructions and isinstance(labels, list) and labels
        if not user_prompt:
            assert len(instructions) == len(labels)
        
        user_prefix = user_prefix or self.template["user_prefix"]
        gpt_prefix = gpt_prefix or self.template["gpt_prefix"]
        
        res = ''
        turns = len(instructions)
        for i in range(turns - 1):
            res += self.template['prompt_history'].format(
                instruction=instructions[i], output=labels[i], user_prefix=user_prefix, gpt_prefix=gpt_prefix, system=self.system if i == 0 else ''
            )
        res += self.template['prompt_no_input'].format(
            instruction=instructions[-1], user_prefix=user_prefix, gpt_prefix=gpt_prefix, system=self.system if turns == 1 else ''
        )
        
        if user_prompt:
            return res
        return f"{res}{labels[-1]}"
                
    @property
    def prompt_bos(self) -> str:
        return self.template["prompt_bos"]

    @property
    def prompt_eos(self) -> str:
        return self.template['prompt_eos']
    
    @property
    def system(self) -> str:
        return self.template["system"]
    
    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[-1].strip()
    
    
if __name__ == '__main__':
    prompter = Prompter('baichuan')
    point1 = {
        'instruction': '三原色是什么？',
        'output': '三原色通常指的是红色、绿色和蓝色（RGB）。它们是通过加色混合原理创建色彩的三种基础颜色。在以发光为基础的显示设备中（如电视、计算机显示器、智能手机和平板电脑显示屏）, 三原色可混合产生大量色彩。其中红色和绿色可以混合生成黄色，红色和蓝色可以混合生成品红色，蓝色和绿色可以混合生成青色。当红色、绿色和蓝色按相等比例混合时，可以产生白色或灰色。\n\n此外，在印刷和绘画中，三原色指的是以颜料为基础的红、黄和蓝颜色（RYB）。这三种颜色用以通过减色混合原理来创建色彩。不过，三原色的具体定义并不唯一，不同的颜色系统可能会采用不同的三原色。'
    }
    print(prompter.generate_prompt(
        point1["instruction"],
        point1.get("input", None),
        point1["output"],
    ))
    
    print("=" * 20)
    
    point2 = {
        'instruction': ['三原色是什么？'],
        'output': ['三原色通常指的是红色、绿色和蓝色（RGB）。它们是通过加色混合原理创建色彩的三种基础颜色。在以发光为基础的显示设备中（如电视、计算机显示器、智能手机和平板电脑显示屏）, 三原色可混合产生大量色彩。其中红色和绿色可以混合生成黄色，红色和蓝色可以混合生成品红色，蓝色和绿色可以混合生成青色。当红色、绿色和蓝色按相等比例混合时，可以产生白色或灰色。\n\n此外，在印刷和绘画中，三原色指的是以颜料为基础的红、黄和蓝颜色（RYB）。这三种颜色用以通过减色混合原理来创建色彩。不过，三原色的具体定义并不唯一，不同的颜色系统可能会采用不同的三原色。']
    }
    print(prompter.generate_prompt(
        point2["instruction"],
        point2.get("input", None),
        point2["output"],
    ))
    
    print("=" * 20)
    
    point3 = {
        "instruction": [
            "我丈夫非常喜欢吃肉。一天，他从肉铺买了一公斤肉，托人带回家。我刚把肉煮好，家里就来了几位朋友，我只好把煮好的肉端出来招待了客人。晚上，丈夫又饿又累地回到家，坐到餐桌前，等着我把煮好的肉端上来美餐一顿，可我只能端给他一碗肉汤饭。他生气地问道：“肉呢?”我无可奈何地回答说：“肉让咱们家的猫吃了。”丈夫有些奇怪地望了望缩在屋角的那只瘦得皮包骨头的猫，愤愤地把猫拿过来，放在了秤上，那只瘦猫不多不少正好一公斤。“喂，老婆你看，这猫才一公斤重，如果这是我们家的猫，那么我买的一公斤肉又到哪儿去了?”他一说完，我们两人都笑了。, 丈夫为什么买了一公斤的肉?",
            "这一公斤肉最后被谁吃了?"
        ],
            "output": [
            "我不知道你的丈夫为什么买了一公斤的肉。在没有更多的信息的情况下，我无法猜测他的动机。然而，根据你所提供的信息，你的丈夫非常喜欢吃肉，所以他可能是为了满足自己的口味而购买肉。",
            "根据你的故事，这一公斤肉最终被你的家庭朋友和一只瘦猫吃了。你在晚上为你的丈夫做肉汤饭时，发现肉已经被你的家庭朋友吃了。你的丈夫很生气，所以你告诉他肉让你家的猫吃了。"
        ]
    }
    output = prompter.generate_prompt(
        point3["instruction"],
        point3.get("input", None),
        point3["output"],
        user_prefix='User',
        gpt_prefix='FinChat-4'
    )
    print(output)
    print("=" * 20)
    print(prompter.get_response(output))
    
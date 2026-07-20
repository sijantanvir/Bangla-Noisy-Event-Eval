import hashlib
import io
import random
import sys


class BengaliErrorGenerator:
    """
    Source: https://github.com/habibsifat/Algorithm-for-Bengali-Error-Dataset-Generation
    """
    
    def __init__(self):
        self.SameClusterDict = {
            'অ' : ['ও'], 'আ' : ['আ'], 'ই':['ই'], 'ঈ':['ই'], 'উ' : ['উ'], 'ঊ' : ['উ'],
            'ঋ' : ['রি'], 'এ' : ['এ'], 'ঐ': ['অই'], 'ও': ['ও'], 'ঔ': ['অউ'],
            'ক': ['ক'], 'খ': ['ক'], 'গ': ['গ'], 'ঘ': ['গ'], 'ঙ': ['ং'],
            'চ': ['চ'], 'ছ': ['চ'], 'জ': ['জ'], 'ঝ': ['জ'], 'ঞ': ['ঞ'],
            'ট' : ['ত'], 'ঠ' : ['ট','ত'], 'ড' : ['দ'], 'ঢ': ['ড','দ'], 'ণ': ['ন'],
            'ত' : ['ত','ট'], 'থ': ['ত','ট'], 'দ' : ['দ','ড'], 'ধ' : ['দ','ড'],
            'ন': ['ন'], 'প' : ['প'], 'ফ' : ['ফ'], 'ব' : ['ব'], 'ভ' : ['ব'], 'ম' : ['ম'],
            'য' : ['জ'], 'র' : ['র'], 'ল' : ['ল'], 'শ' : ['শ'], 'ষ' : ['শ'], 'স' : ['স'],
            'হ': ['হ'], 'ড়' : ['র'], 'ঢ়': ['র'], 'য়': ['য়'],
            'ৎ' : ['ত'], 'ং' : ['ঙ'], 'ঃ': ['হ'], 'ঁ' : [''],
            'ি': ['ি'], 'ী' : ['ি'], 'ে':['ে'], 'ৈ':['ই'], 'ো' : ['ো'],
            'ৌ':['উ'], 'ৃ' : ['রি'], 'ূ' : ['ু'], 'ু' : ['ূ']
        }

        self.ReplaceDict = {
            'ক' : ['ল','য'], 'খ' : ['কগ','কজ','লহ','ঝ'], 'গ' : ['ফ','হ'], 'ঘ' : ['ফগ','হজ'],
            'ঙ' : ['ব','ম'], 'চ' : ['ভ','চজ','চগ','ভহ'], 'ছ' : ['ভ','চজ','চগ','ভহ'],
            'জ' : ['ক','হ'], 'ঝ' : ['কজ','হগ'], 'ট' : ['র'], 'ঠ' : ['তজ','তগ','রহ'],
            'ড' : ['স','ফ'], 'ঢ' : ['দজ','দ্গ','শ','ফহ'], 'ণ' : ['ব','ম'], 'ত' : ['র'],
            'থ' : ['তজ', 'তগ', 'রহ'], 'দ' : ['স', 'ফ'], 'ধ' : ['দজ', 'দ্গ', 'শ' ,'ফহ'],
            'ন' : ['ব','ম'], 'প' : ['ও','ো'], 'ফ' : ['দ','গ'], 'ব' : ['ভ','ন'],
            'ভ' : ['ব','চ'], 'ম' : ['ন'], 'য' : ['হ','ক'], 'র' : ['এ','ে','ত'], 'ল' : ['ক'],
            'শ' : ['সজ','সগ','আহ','ঢ'], 'ষ' : ['সজ','সগ','আহ','ঢ'], 'স' : ['আ','া','দ'],
            'হ' : ['গ','য'], 'য়' : ['ত','উ','ু'], 'ড়' : ['এ','ে','ত'], 'ঢ়' : ['এ','ে','ত'],
            'ৎ' : ['র'], 'ং' : ['ব','ম'], 'ঃ' : ['গ','য'], 'অ' : ['প','ি','ই'], 'আ' : ['স'],
            'ই' : ['উ','অ'], 'ঈ' : ['উ','অ'], 'উ' : ['ই','ি'], 'ঊ' : ['ই','ি'],
            'এ' : ['ও','র'], 'ও' : ['প','ই'],
            'া' : ['স'], 'ি' : ['উ','অ'], 'ো' : ['প','ি'], 'ৌ' : [''], 'ে' : ['ো','র'], 'ৈ' : ['']
        }

        self.JuktakkhorList = [' ক্ট ' , ' ক্ক ' , ' ক্ত ' , ' ক্য ' , ' ক্র ' , ' ক্ল ' , ' ক্ষ ' , ' ক্ষ্ণ ' , ' ক্ষ্ম ' , ' ক্ষ্য ' , ' ক্স ' , ' খ্র ' , ' গ্ধ ' , ' গ্ধ্য ' , ' গ্ন ' , ' গ্ন্য ' , ' গ্ব ' , ' গ্র ' , ' গ্র্য ' , ' গ্ল ' , ' ঘ্ন ' , ' ঘ্র ' , ' ঙ্ক্য ' , ' ঙ্গ্য ' , ' চ্চ ' , ' চ্ছ্ব ' , ' চ্য ' , ' জ্জ ' , ' জ্জ্ব ' , ' ট্ট ' , ' জ্ব ' , ' জ্য ' , ' জ্র ' , ' ট্য ' , ' ট্র ' , ' ড্ড ' , ' ড্র ' , ' ণ্ট ' , ' ণ্ঠ ' , ' ণ্ড ' , ' ণ্ণ ' , ' ণ্য ' , ' ৎক ' , ' ৎখ ' , ' ত্ত ' , ' ত্ত্ব ' , ' ত্ত্য ' , ' ত্ন ' , ' ৎপ ' , ' ত্ব ' , ' ত্ম ' , ' ত্ম্য ' , ' ত্য ' , ' ত্র ' , ' ত্র্য ' , ' ৎস ' , ' দ্ঘ ' , ' দ্দ ' , ' দ্ধ ' , ' দ্ব ' , ' দ্ভ ' , ' দ্ভ্র ' , ' দ্ম ' , ' দ্য ' , ' দ্র ' , ' দ্র্য ' , ' ধ্ব ' , ' ধ্য ' , ' ধ্র ' , ' ন্ট ' , ' ন্ট্র ' , ' ন্ঠ ' , ' ন্ড ' , ' ন্ড্র ' , ' ন্ত ' , ' ন্ত্ব ' , ' ন্ত্য ' , ' ন্ত্র ' , ' ন্ত্র্য ' , ' ন্থ ' , ' ন্দ ' , ' ন্দ্ব ' , ' ন্দ্র ' , ' ন্ধ ' , ' ন্ন ' , ' ন্য ' , ' প্ট ' , ' প্ত ' , ' প্ন ' , ' প্প ' , ' প্য ' , ' ব্দ ' , ' ব্ধ ' , ' ব্ব ' , ' ব্র ' , ' ভ্য ' , ' ভ্র ' , ' ম্প্র ' , ' ম্ব ' , ' ম্ম ' , ' ম্য ' , ' ম্র ' , ' য্য ' , ' র্ক ' , ' র্গ্য ' , ' র্ঘ্য ' , ' র্জ্য ' , ' র্থ্য ' , ' র্ব্য ' , ' র্খ ' , ' র্গ ' , ' র্ঘ ' , ' র্চ ' , ' র্ছ ' , ' র্জ ' , ' র্ঝ ' , ' র্ট ' , ' র্ড ' , ' র্ণ ' , ' র্ত ' , ' র্থ ' , ' র্দ ' , ' র্দ্ব ' , ' র্দ্র ' , ' র্ধ ' , ' র্ধ্ব ' , ' ল্ট ' , ' ল্ড ' , ' ল্প ' , ' ল্ম ' , ' ল্য ' , ' ল্ল ' , ' শ্ছ ' , ' শ্ন ' , ' শ্ব ' , ' শ্র ' , ' শ্ল ' , ' ষ্ক ' , ' ষ্ক্র ' , ' ষ্ট ' , ' ষ্ট্য ' , ' ষ্ট্র ' , ' ষ্ঠ ' , ' ষ্প ' , ' স্ট ' , ' স্ট্র ' , ' স্ত্র ' , ' স্ত ' , ' স্থ্য ' , ' স্য ' , ' স্র ' , ' স্ল ' , ' হ্র ' , ' হ্ল ' , ' গ্ম ' , ' ম্ভ ' , ' স্ম ' , ' ঙ্ঘ ' , ' ঙ্ক্ষ ' , ' ঙ্খ ' , ' ঙ্ক ' , ' ঙ্‌ক্ত ' , ' ঞ্চ ' , ' ঞ্ছ ' , ' ঞ্জ ' , ' হ্ন ' , ' ঙ্গ ' , ' জ্ঞ ' , ' গ্য ' , ' চ্ছ ' , ]

        self.InsertDict = {
            'ক': ['ল'], 'খ': ['গ'], 'গ' : ['ফ','হ'], 'ঘ' : ['জ'], 'ঙ' : ['ম'],
            'চ' : ['জ'], 'ছ' : ['জ'], 'জ' : ['ক'], 'ঝ' : ['গ'], 'ট' : ['র'], 'ঠ' : ['জ'],
            'ড' : ['স'], 'ঢ' : ['জ'], 'ণ' : ['ম'], 'ত' : ['র'], 'থ' : ['জ'], 'দ' : ['স'],
            'ধ' : ['জ'], 'ন' : ['ম'], 'প' : [''], 'ফ' : ['দ'], 'ব' : ['ন'], 'ম' : [''],
            'য' : ['ক'], 'র' : ['ে'], 'ল' : [''], 'শ' : ['জ'], 'ষ' : ['জ'], 'স' : ['া'],
            'হ' : ['জ'], 'য়' : ['ু'], 'ড়' : ['ে'], 'ঢ়' : ['ে'], 'ৎ' : ['র'], 'ং' : ['ম'],
            'ঃ' : ['জ'], 'অ' : ['প'], 'আ' : [''], 'ই' : ['অ'], 'ঈ' : ['অ'], 'উ' : ['ি'],
            'ঊ' : ['ি'], 'ঋ' : ['ো'], 'এ' : ['ও'], 'ঐ' : ['ু'], 'ও' : ['প'], 'ঔ' : ['য়'],
        }

        self.character = ['অ','আ','ই','ঈ','উ','ঊ','এ','ঐ','ও','ঔ','ঋ','ঃ','ং','ৎ',
                          'ক','খ','গ','ঘ','ঙ','চ','ছ','য','জ','ঝ','ট','ঠ','ড','র','ড়','ঢ়',
                          'ণ','ত','থ','দ','ধ','ন','প','ফ','ব','ভ','ম','স','শ','ষ','য়','হ']

        self.nlist = []
        self.NewWord = ""

    def SameClusterFun(self, c):
        if c in self.SameClusterDict:
            c2 = random.choice(self.SameClusterDict[c])
            print(c2, end='')
        else:
            print(c, end='')

    def ReplacementFun(self, c):
        if c in self.ReplaceDict:
            c2 = random.choice(self.ReplaceDict[c])
            print(c2, end='')
        else:
            print(c, end='')

    def JuktoBorno(self, word, pos):
        if word[pos] == 'জ' and word[pos+2] == 'ঞ':
            if pos == 0:
                print("গ", end='')
            else:
                print("জ্ঞ", end='')
        elif word[pos] == 'গ' and word[pos+2] == 'য':
            if pos == 0:
                print("গা", end='')
            else:
                print("জ্ঞ", end='')
        elif word[pos] == 'চ' and word[pos+2] == 'ছ':
            r = random.random()
            if r < .5:
                print("ছছ", end='')
            else:
                print("ছ", end='')
        elif word[pos+2] == 'য':
            if pos+2 == len(word):
                print(word[pos], end='')
                print(word[pos], end='')
            else:
                r = random.random()
                if r < .5:
                    print(word[pos], end='')
                    print("া", end='')
                else:
                    print("ে", end='')
                    print(word[pos], end='')
        elif word[pos] == 'স' and word[pos+2] == 'ম':
            self.SameClusterFun(word[pos])
        elif word[pos] == 'দ' and word[pos+2] == 'ম':
            self.SameClusterFun(word[pos])
            self.SameClusterFun(word[pos])
        elif word[pos] == 'ম':
            self.SameClusterFun(word[pos+2])
        elif word[pos+2] == 'ম':
            self.SameClusterFun(word[pos])
        elif word[pos] == 'ব':
            print(word[pos+2], end='')
        elif word[pos+2] == 'ব':
            print(word[pos], end='')
        elif word[pos] == 'র':
            print(word[pos], end='')
            self.SameClusterFun(word[pos+2])
        elif word[pos+2] == 'র':
            self.SameClusterFun(word[pos])
            print(word[pos+2], end='')
        elif word[pos] == 'ক' and word[pos+2] == 'ষ':
            if pos == 0:
                print("খ", end='')
            else:
                print("ক্ক", end='')
        elif word[pos+2] == 'ঙ':
            if word[pos+3] == "া":
                self.SameClusterFun(word[pos])
                print("ঙ্গা", end='')
            else:
                self.SameClusterFun(word[pos])
                print("ং", end='')
        elif word[pos] == 'ঙ':
            print("ং", end='')
            self.SameClusterFun(word[pos+2])
        elif word[pos] == 'ঞ':
            print("ঞ", end='')
            self.SameClusterFun(word[pos+2])
        elif word[pos] == 'হ' and word[pos+2] == 'ন':
            print("ন্ন", end='')
        elif word[pos] == 'ন' and word[pos+2] == 'ন':
            print("হ্ন", end='')
        elif word[pos] == word[pos+2]:
            self.SameClusterFun(word[pos+2])
            self.SameClusterFun(word[pos+2])
        elif word[pos] in ['ল','ত','থ','দ','ধ','ট','ঠ','স','শ','ষ']:
            print(word[pos], end='')
            self.SameClusterFun(word[pos+2])
        elif word[pos+2] in ['ল','ত','থ','দ','ধ','ট','ঠ','স','শ','ষ']:
            self.SameClusterFun(word[pos])
            print(word[pos+2], end='')
        elif word[pos] in ['ন','ণ'] or word[pos+2] in ['ন','ণ']:
            self.SameClusterFun(word[pos])
            self.SameClusterFun(word[pos+2])
        else:
            print(word[pos], end='')
            print(word[pos+1], end='')
            print(word[pos+2], end='')

    def MakeError(self, word):
        self.nlist = []
        pos = 0
        flag1 = 0
        flag2 = 0
        flag3 = 0
        while pos < len(word):
            if flag1 == 1 and flag2 == 1 and flag3 == 1:
                print(word[pos], end='')
                self.nlist.append(word[pos])
                pos += 1
            else:
                if pos+1 < len(word) and word[pos+1] == '্':
                    r = random.random()
                    if r <= .70 and flag1 == 1:
                        print(word[pos], end='')
                        self.nlist.append(word[pos])
                        print(word[pos+1], end='')
                        self.nlist.append(word[pos+1])
                        print(word[pos+2], end='')
                        self.nlist.append(word[pos+2])
                        pos = pos+3
                    else:
                        flag1 = 1
                        self.JuktoBorno(word, pos)
                        pos = pos+3
                else:
                    r = random.random()
                    if r <= .2:
                        flag2 = 1
                        self.SameClusterFun(word[pos])
                    elif r > .5 and r <= .7:
                        if flag3 == 1:
                            print(word[pos], end='')
                            self.nlist.append(word[pos])
                        else:
                            flag3 = 1
                            self.ReplacementFun(word[pos])
                    else:
                        print(word[pos], end='')
                        self.nlist.append(word[pos])
                    pos += 1

    def lenth(self, words):
        l = 0
        for i in words:
            if i in self.character:
                l += 1
        return l

    def insert(self, word2):
        valid = [c for c in word2 if c in self.InsertDict and self.InsertDict[c] != ['']]
        if not valid:
            return '', 0
        c = random.choice(valid)
        position = word2.index(c)
        c2 = random.choice(self.InsertDict[c])
        return c2, position+1

    def insertChar(self, mystring, position, chartoinsert):
        return mystring[:position] + chartoinsert + mystring[position:]

    def corrupt_word(self, word):
        if len(word) < 2:
            return word

        stable_seed = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16) % (10**8)
        random.seed(stable_seed)

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        self.MakeError(word)
        pass1 = sys.stdout.getvalue()
        sys.stdout = old_stdout

        pass1 = pass1.strip() if pass1.strip() else word

        if self.lenth(pass1) > 3:
            element, index = self.insert(pass1)
            if element:
                return self.insertChar(pass1, index, element)

        return pass1

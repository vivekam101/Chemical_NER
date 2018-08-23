import re

chemicals_dict = {}
def add_to_dict(list_):
    if list_:
        for name in list_:
            if name in chemicals_dict:
                chemicals_dict[name.lower()] += 1
            else: 
                chemicals_dict[name.lower()] = 1
    return chemicals_dict

prefix = []
with open("chemicals_.txt") as f:
    while True:
        chunk = f.read(4096)
        if not chunk:
            break
        #temp = re.findall(r"(\w+ \w+ate)",chunk)
        chemicals_dict.update(add_to_dict(re.findall(r"(\w+ \w+ate)",chunk.lower())))
        chemicals_dict.update(add_to_dict(re.findall(r"(\w+ \w+ide)",chunk.lower())))
        chemicals_dict.update(add_to_dict(re.findall(r"(\w+ \w+ina)",chunk.lower())))
        chemicals_dict.update(add_to_dict(re.findall(r"(\w+ \w+ose)",chunk.lower())))
        chemicals_dict.update(add_to_dict(re.findall(r"(\w+ine)",chunk.lower())))

chem = []
for k,v in chemicals_dict.items():
    if v>3:
        chem.append(k)
chem1 = []
for value in chem:
    if(len(value.split(" ")[0])>3):
        chem1.append(value)

print(chem1)

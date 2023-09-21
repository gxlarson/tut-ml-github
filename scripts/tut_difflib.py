from difflib import SequenceMatcher

s1 = "this is the second document"
s2 = "this um is the second document"

d = SequenceMatcher(a=s1.split(), b=s2.split())

r = d.ratio()
print(r)

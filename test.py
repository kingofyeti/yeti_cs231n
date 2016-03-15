import sys,os

os.system('git commit -am \"' + sys.argv[1] + '\"' )
os.system('git add -A')
os.system('git push -u')


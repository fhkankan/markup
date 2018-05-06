# Git global setup

```
git config --global user.name "付杭"
git config --global user.email "ex-fuhang001@pingan.com.cn"
```

# Create a new repository

```
git clone git@www.paicrobot.com:EX-FUHANG001/robotQA.git
cd robotQA
touch README.md
git add README.md
git commit -m "add README"
git push -u origin master
```

 # Existing folder

```
cd existing_folder
git init
git remote add origin git@www.paicrobot.com:EX-FUHANG001/robotQA.git
git add .
git commit -m "Initial commit"
git push -u origin master
```

# Existing Git repository

```
cd existing_repo
git remote rename origin old-origin
git remote add origin git@www.paicrobot.com:EX-FUHANG001/robotQA.git
git push -u origin --all
git push -u origin --tags
```




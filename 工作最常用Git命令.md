### 使用频次最高的Git命令
##### clone代码到本地
```bash
git clone xxx.git
拉取最新代码，先git fetch更新所有远程仓库，再用git pull从最新的仓库拉取代码
git fetch
git pull
```
##### 切换分支，实际工作中经常要切换到分支进行开发
```bash
1、直接切换，但当本地有改动时无法切换，一般采用流程2
git checkout $branch_name
2、缓存本地改动再切换
git stash save "xxxx"
git stash list
git checkout $branch_name
切换回去
git checkout $raw_branch_name
释放缓存，$x是多少用git stash list查看  
git stash apply stash@{$x} or git stash pop stash@{x}
```

##### push代码到远程仓库一般流程
```bash
查看当前代码改动
git status
git diff $file_path
确认改动正确后，添加改动到本地git，为了防止出错，一般逐个添加
git add $file_path
git commit -m "xxxx"
推送到默认的远程分支
git push
也可推送到指定的远程分支，如不存在，则会新建远程分支，因此常用来新建分支
git push origin $local-branch-name:$remote-branch-name
```
### 其他使用频次较低的Git命令
```bash
新建本地分支
git checkout -b $branch_name
从其他分支更新指定文件，会直接覆盖原内容
git checkout $branch_name -- file_path
从其他分支pull代码
git pull origin $remote_branch:$local_branch
删除本地分支
git branch -d $branch_name
缓存单个文件
git stash -- temp.c
缓存多个文件
git stash push a_file b_file c_file ...
查看所有分支
git branch -a
复原文件
git restore --staged filepath
撤销上次commit，但commit内容是保留在缓存中的
git reset --soft HEAD^
丢弃所有uncommit的changes
git reset --hard $版本号
查看上次git pull的时间
date  -r .git/FETCH_HEAD
比较两个分支或者commit不同
git diff a b --stat
比较具体文件
git diff a b filename
查看被修改的文件列表
git diff --name-only a b
用于查看当前分支以及分支之间的关系和跟踪情况
git branch -vv
查看最近两次两次提交
git reflog -n 2
合并指定commit代码
git cherry-pick $commit_id
```
### Git Fork Workflow with Specific Repositories

This guide outlines how to manage forks and synchronize changes for our specific repos.

#### Cloning the Fork

First, clone your fork of the original repository. In this case, your forked repository is `Luka-R-Lukacevic/prosperity2`.

```bash
git clone https://github.com/Luka-R-Lukacevic/prosperity2.git
cd prosperity2
```

#### Setting Up Remotes
Set up the original repository as a remote to easily sync changes. The original repository is file-acomplaint/prosperity2.

```bash
# Add the original repository as a remote called "upstream"
git remote add upstream https://github.com/file-acomplaint/prosperity2.git

# Verify the new remote named "upstream" is added
git remote -v
```

#### Syncing Your Fork
Keep your fork (Luka-R-Lukacevic/prosperity2) up-to-date with the original repository (file-acomplaint/prosperity2).

```bash
# Fetch changes from the original repository
git fetch upstream

# Merge changes from the original repository into your local main branch
git checkout main
git merge upstream/main

# Push updates to your fork on GitHub
git push origin main
```

#### Making Changes and Pushing to Your Fork
After syncing, you can make changes to your fork and push them as follows:

```bash
# After making changes
git add .
git commit -m "Describe your changes here"
git push origin main
```
#### Contributing to the Original Repository
To contribute your changes from your fork (Luka-R-Lukacevic/prosperity2) back to the original repository (file-acomplaint/prosperity2), create a pull request from your fork on GitHub.

1. Go to your fork on GitHub.
2. Click "Pull Requests" > "New Pull Request".
3. Choose the original repository's main branch as the base and your fork's branch where you made changes as the compare.
4. Fill in the pull request details and create it.
This workflow allows you to keep your fork up-to-date and contribute changes back to the original project.

This workflow allows you to keep your fork up-to-date and contribute changes back to the original project.

```vbnet

Make sure to replace `main` with the correct branch name if it's different in your repositories.
```

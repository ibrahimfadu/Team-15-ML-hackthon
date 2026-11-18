# GitHub workflow for three members on one development laptop

## One-time setup by Yash (repository owner)

1. Create a **public** GitHub repository named `PathSnap-Team15-Section5D`.
2. Add Deeraj S and Ibrahim Khaleel as collaborators with write access.
3. In CMD from this project folder:

```cmd
git init
git branch -M main
git add .
git commit -m "chore: initialize PathSnap Team 15 project"
git remote add origin https://github.com/<YASH_GITHUB_USERNAME>/PathSnap-Team15-Section5D.git
git push -u origin main
```

## Feature branches

Each member works on their own branch and commits only work they actually implemented/reviewed:

```cmd
git switch -c feature/yash-data
```

```cmd
git switch main
git pull
git switch -c feature/deeraj-models
```

```cmd
git switch main
git pull
git switch -c feature/ibrahim-decoder-ui
```

After making that member's changes:

```cmd
git add <their-files>
git commit -m "feat: describe the actual change"
git push -u origin <branch-name>
```

Open a Pull Request on GitHub from the feature branch into `main`. Have another teammate review it before merging.

## Shared laptop author identity

The person who actually made a change should make its commit under their own verified GitHub email/identity. Git author metadata and GitHub push authentication are separate: `git config user.name`/`user.email` set the commit author; GitHub CLI or Git Credential Manager authenticates the push. Each collaborator should authenticate their own account when pushing. Do not share personal access tokens and do not fabricate commits or authorship.

## Before submission

```cmd
git status
git log --oneline --decorate --all
git remote -v
```

Make the repository public, verify the README works on a clean clone, and ensure no `.env`, credentials, or restricted raw data are committed.

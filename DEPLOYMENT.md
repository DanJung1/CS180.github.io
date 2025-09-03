# 🚀 GitHub Pages Deployment Guide

Follow these steps to deploy your CS180 Photography Project website on GitHub Pages.

## Step 1: Create a GitHub Repository

1. **Go to GitHub.com** and sign in to your account
2. **Click the "+" icon** in the top right corner
3. **Select "New repository"**
4. **Fill in the details**:
   - Repository name: `cs180-photography-project` (or any name you prefer)
   - Description: `CS180 Photography Project - Becoming Friends with Your Camera`
   - Make it **Public** (required for free GitHub Pages)
   - **Don't** initialize with README (we already have one)
5. **Click "Create repository"**

## Step 2: Upload Your Project Files

### Option A: Using GitHub Desktop (Recommended for beginners)

1. **Download GitHub Desktop** from [desktop.github.com](https://desktop.github.com/)
2. **Clone the repository** to your computer
3. **Copy all project files** into the repository folder:
   - `index.html`
   - `styles.css`
   - `script.js`
   - `README.md`
   - `images/` folder
4. **Commit and push** your changes

### Option B: Using Git Commands

1. **Open Terminal/Command Prompt**
2. **Navigate to your project folder**
3. **Run these commands**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: CS180 Photography Project"
   git branch -M main
   git remote add origin https://github.com/[your-username]/[repo-name].git
   git push -u origin main
   ```

### Option C: Direct Upload on GitHub

1. **Go to your repository** on GitHub
2. **Click "Add file"** → "Upload files"
3. **Drag and drop** all your project files
4. **Click "Commit changes"**

## Step 3: Enable GitHub Pages

1. **Go to your repository** on GitHub
2. **Click "Settings"** tab
3. **Scroll down** to "Pages" section (or click "Pages" in the left sidebar)
4. **Under "Source"**, select **"Deploy from a branch"**
5. **Choose branch**: `main` (or `master`)
6. **Choose folder**: `/ (root)`
7. **Click "Save"**

## Step 4: Wait for Deployment

- **GitHub will show**: "Your site is being built"
- **Wait 2-5 minutes** for the first deployment
- **You'll see**: "Your site is published at [URL]"

## Step 5: Access Your Website

Your website will be available at:
```
https://[your-username].github.io/[repository-name]
```

**Example**: If your username is `johndoe` and repo is `cs180-photography-project`:
```
https://johndoe.github.io/cs180-photography-project
```

## Step 6: Test Your Website

1. **Visit your website URL**
2. **Check all sections** load properly
3. **Test photo uploads** by clicking placeholders
4. **Test mobile responsiveness** by resizing browser
5. **Verify navigation** works smoothly

## 🔄 Updating Your Website

### Every time you make changes:

1. **Upload new files** to your repository
2. **Commit changes** with a descriptive message
3. **GitHub Pages automatically updates** (usually within 2-5 minutes)

### Example update process:
```bash
git add .
git commit -m "Added new photos and updated gallery"
git push origin main
```

## 🚨 Common Issues & Solutions

### Website Not Loading
- **Check repository is public**
- **Verify GitHub Pages is enabled**
- **Wait longer for first deployment**
- **Check for build errors** in repository Actions tab

### Photos Not Displaying
- **Ensure image files are uploaded**
- **Check file paths are correct**
- **Verify image formats are supported** (JPG, PNG, GIF)

### Changes Not Appearing
- **Wait 2-5 minutes** for GitHub Pages to update
- **Clear browser cache** (Ctrl+F5 or Cmd+Shift+R)
- **Check repository** has latest changes

## 📱 Testing Checklist

- [ ] **Desktop**: All sections visible, navigation works
- [ ] **Tablet**: Responsive layout, touch interactions
- [ ] **Mobile**: Hamburger menu, proper scaling
- [ ] **Photos**: Upload functionality works
- [ ] **Animations**: Smooth scrolling and effects
- [ ] **Links**: All navigation links work
- [ ] **Performance**: Fast loading on different devices

## 🎯 Final Steps

1. **Test everything thoroughly**
2. **Take screenshots** of your working website
3. **Submit the URL** to the class gallery form
4. **Print as PDF** for Gradescope submission
5. **Celebrate** your beautiful photography website! 🎉

## 📞 Need Help?

- **GitHub Help**: [help.github.com](https://help.github.com/)
- **GitHub Pages Docs**: [pages.github.com](https://pages.github.com/)
- **Course Staff**: Check your CS180 course materials
- **Classmates**: Collaborate and help each other

---

**Good luck with your deployment! Your website will look amazing! ✨**

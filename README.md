# CS180 Photography Project Website

A beautiful, modern GitHub Pages website for showcasing your CS180 "Becoming Friends with Your Camera" photography project.

## 🌟 Features

- **Modern Design**: Clean, aesthetic interface with smooth animations
- **Responsive Layout**: Works perfectly on desktop, tablet, and mobile devices
- **Interactive Photo Uploads**: Click on photo placeholders to upload your images
- **Smooth Navigation**: Fixed navigation bar with smooth scrolling
- **Beautiful Animations**: Fade-in effects and hover animations
- **Photo Gallery**: Organized display of all your project photos
- **Mobile-Friendly**: Hamburger menu for mobile devices

## 📸 Project Sections

### Part 1: Selfie Comparison
- **The Wrong Way**: Close-up selfie showing distortion
- **The Right Way**: Stepped back + zoomed in for natural proportions
- **Explanation**: Why perspective distortion occurs and how to fix it

### Part 2: Architectural Perspective
- **Zoomed In**: Compressed, flattened perspective
- **No Zoom**: Natural depth and perspective
- **Explanation**: How zoom affects urban scene perception

### Part 3: Dolly Zoom Effect
- **Interactive GIF Display**: Showcase your animated dolly zoom
- **Step-by-Step Guide**: How to create the effect
- **Technical Explanation**: Understanding the "Vertigo shot"

### Photo Gallery
- **Grid Layout**: Beautiful display of all project photos
- **Hover Effects**: Interactive elements for better engagement

## 🚀 Getting Started

### Option 1: GitHub Pages (Recommended)

1. **Fork or Clone this repository**
   ```bash
   git clone [your-repo-url]
   cd cs180-photography-project
   ```

2. **Enable GitHub Pages**
   - Go to your repository on GitHub
   - Click "Settings" → "Pages"
   - Select "Deploy from a branch"
   - Choose "main" branch and "/ (root)" folder
   - Click "Save"

3. **Your website will be available at**: `https://[your-username].github.io/[repo-name]`

### Option 2: Local Development

1. **Clone the repository**
   ```bash
   git clone [your-repo-url]
   cd cs180-photography-project
   ```

2. **Open in your browser**
   - Simply open `index.html` in your web browser
   - Or use a local server:
     ```bash
     # Using Python 3
     python -m http.server 8000
     
     # Using Node.js
     npx serve .
     ```

## 📁 File Structure

```
cs180-photography-project/
├── index.html          # Main HTML file
├── styles.css          # CSS styling and animations
├── script.js           # JavaScript functionality
├── README.md           # This file
└── images/             # Create this folder for your photos
    ├── selfie-wrong.jpg
    ├── selfie-right.jpg
    ├── architecture-zoomed.jpg
    ├── architecture-natural.jpg
    ├── dolly-zoom.gif
    └── gallery-photos/
```

## 🖼️ Adding Your Photos

### Method 1: Interactive Upload (Recommended)
1. **Click on any photo placeholder** on the website
2. **Select your image file** from your computer
3. **The image will automatically replace the placeholder**
4. **Click the × button** to remove and re-upload if needed

### Method 2: Direct File Replacement
1. **Place your photos** in the `images/` folder
2. **Update the HTML** to reference your image files
3. **Replace photo placeholders** with `<img>` tags

### Recommended Image Specifications
- **Format**: JPG for photos, GIF for animations
- **Size**: 800x600px minimum, 1920x1080px maximum
- **File Size**: Keep under 5MB for optimal loading
- **Quality**: High quality for best visual impact

## 🎨 Customization

### Colors
Edit `styles.css` to change the color scheme:
```css
:root {
    --primary-color: #2563eb;      /* Main blue */
    --secondary-color: #667eea;    /* Gradient start */
    --accent-color: #764ba2;       /* Gradient end */
}
```

### Fonts
Change fonts by updating the Google Fonts link in `index.html`:
```html
<link href="https://fonts.googleapis.com/css2?family=YourFont:wght@300;400;500;600;700&display=swap" rel="stylesheet">
```

### Content
- **Update project information** in the HTML
- **Modify section descriptions** to match your specific project
- **Add your name** in the footer section

## 📱 Mobile Optimization

The website is fully responsive and includes:
- **Touch-friendly navigation**
- **Optimized layouts** for small screens
- **Fast loading** on mobile networks
- **Proper viewport scaling**

## 🔧 Technical Details

- **HTML5**: Semantic markup for accessibility
- **CSS3**: Modern styling with Flexbox and Grid
- **JavaScript ES6+**: Interactive functionality
- **Font Awesome**: Icon library for visual elements
- **Google Fonts**: Beautiful typography

## 🌐 Browser Support

- **Chrome**: 90+
- **Firefox**: 88+
- **Safari**: 14+
- **Edge**: 90+

## 📝 Submission Requirements

### For GitHub Pages Submission:
1. **Deploy your website** using GitHub Pages
2. **Submit the URL** to the class gallery form
3. **Ensure all photos are visible** and properly displayed

### For PDF Submission:
1. **Print your webpage** as PDF
2. **Include the URL** in the header
3. **Submit to Gradescope** with entry code: VWX283

## 🚨 Troubleshooting

### Photos Not Displaying
- Check file paths and names
- Ensure images are in the correct format
- Verify file sizes are reasonable

### Website Not Loading
- Check GitHub Pages settings
- Verify repository is public
- Wait a few minutes for deployment

### Mobile Issues
- Test on different devices
- Check responsive design settings
- Verify touch interactions work

## 📚 Additional Resources

- [GitHub Pages Documentation](https://pages.github.com/)
- [CS180 Course Information](https://cs180.stanford.edu/)
- [Photography Tips](https://digital-photography-school.com/)
- [GIF Creation Tools](https://ezgif.com/)

## 🤝 Contributing

Feel free to:
- **Report bugs** or issues
- **Suggest improvements** to the design
- **Share your photos** and experiences
- **Help other students** with their projects

## 📄 License

This project is created for educational purposes as part of CS180: Intro to Computer Vision and Computational Photography.

---

**Good luck with your photography project! 📸✨**

*Remember: Aesthetics matter! Make your photos beautiful and your website visually appealing.*

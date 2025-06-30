window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]], // Default inline delimiters for LaTeX
    displayMath: [["\\[", "\\]"]], // Default block delimiters for LaTeX
    processEscapes: true, // Process escaped characters
    processEnvironments: true // Process LaTeX environments (e.g., \begin{align})
  },
  options: {
    ignoreHtmlClass: ".*|", // Ignore anything that looks like math in these classes
    processHtmlClass: "arithmatex" // Only process elements with this class (from pymdownx.arithmatex)
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise()
})
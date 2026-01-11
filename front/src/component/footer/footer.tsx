import React from "react";
import "./footer.css";

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-container">
        <div className="footer-brand">
          <span className="footer-icon">🧠</span>
          <span className="footer-name">Neurox</span>
        </div>

        <p className="footer-tagline">
          Open-source AI image processing • Built by Avo
        </p>

        <div className="footer-badges">
          <span className="badge">🔒 Secure</span>
          <span className="badge">⚡ Fast</span>
          <span className="badge">🚫 No signup</span>
        </div>

        <a
          href="https://github.com/AvoApanian/NeuroX"
          target="_blank"
          rel="noopener noreferrer"
          className="footer-github"
        >
          ⭐ View on GitHub
        </a>

        <p className="footer-copyright">
          © 2026 Neurox • Open Source
        </p>
      </div>
    </footer>
  );
}

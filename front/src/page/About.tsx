import React from 'react';
import './../pageCss/About.css';

export default function AboutPage() {
  const features = [
    {
      icon: "🧠",
      title: "AI-Powered Enhancement",
      description: "Transform low-quality images into stunning high-resolution results using advanced neural networks."
    },
    {
      icon: "🖼️",
      title: "Controlled Degradation",
      description: "Intentionally reduce image quality for specific use cases with precise control."
    },
    {
      icon: "⚡",
      title: "Fast Processing",
      description: "Automated pipeline ensures quick turnaround times without compromising quality."
    },
    {
      icon: "🛡️",
      title: "Security First",
      description: "Multi-layer validation prevents malicious uploads. Your safety is our priority."
    }
  ];

  const techStack = [
    { name: "React", category: "Frontend" },
    { name: "FastAPI", category: "Backend" },
    { name: "Express", category: "Backend" },
    { name: "PyTorch", category: "AI Framework" },
    { name: "PostgreSQL", category: "Database" }
  ];

  const securityFeatures = [
    "Only image files are accepted as input",
    "File type validation enforced on the frontend",
    "Binary signatures verified on the backend",
    "Strict file size and memory limits prevent RAM exhaustion",
    "Automatic rejection of suspicious or oversized files"
  ];

  return (
    <div className="about-page">
      <div className="about-background-overlay" />

      <div className="about-container">
        {/* Hero Section */}
        <div className="hero-section">
          <h1 className="hero-title">About Neurox</h1>
          <p className="hero-subtitle">
            Advanced AI-powered image processing designed for simplicity, security, and accessibility
          </p>
        </div>

        <div className="mission-card">
          <h2 className="section-title">Our Vision</h2>
          <p className="mission-text">
            Neurox is a solo-developed personal project built with a strong focus on performance, security, and ease of use. 
            The goal is to deliver a frictionless, privacy-first image processing platform powered by modern AI technologies. 
            No accounts, no passwords, no emails required — just pure, secure image enhancement at your fingertips.
          </p>
        </div>

        <div className="features-section">
          <h2 className="section-title center">Core Features</h2>
          <div className="features-grid">
            {features.map((feature, index) => (
              <div key={index} className="feature-card">
                <div className="feature-icon-wrapper">
                  <span className="feature-emoji">{feature.icon}</span>
                </div>
                <h3 className="feature-title">{feature.title}</h3>
                <p className="feature-description">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="security-section">
          <div className="security-header">
            <span className="security-emoji">🛡️</span>
            <h2 className="section-title">Security by Design</h2>
          </div>
          <p className="security-intro">
            Security is a core principle of Neurox, not an afterthought. Every upload goes through multiple validation layers:
          </p>
          <ul className="security-list">
            {securityFeatures.map((item, index) => (
              <li key={index} className="security-item">
                <span className="security-bullet" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="tech-section">
          <h2 className="section-title center">Technology Stack</h2>
          <div className="tech-grid">
            {techStack.map((tech, index) => (
              <div key={index} className="tech-card">
                <div className="tech-category">{tech.category}</div>
                <div className="tech-name">{tech.name}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
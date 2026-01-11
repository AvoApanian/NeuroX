import React from "react";
import { Link, useNavigate } from "react-router-dom";
import "./Navbar.css";
import logo from "./../../image/logo.webp";

const Navbar: React.FC = () => {
  const navigate = useNavigate();

  const handleLogoClick = () => {
    // Recharge la page comme F5
    window.location.href = "/";
  };

  return (
    <nav className="navbar">
      <div className="left">
        <img 
          src={logo} 
          alt="logo" 
          className="logo" 
          onClick={handleLogoClick}
          style={{ cursor: 'pointer' }}
        />
      </div>
      <ul className="right">
        <li><Link to="/">Home</Link></li>
        <li><Link to="/photo">Photo</Link></li>
        <li><Link to="/about">About</Link></li>
      </ul>
    </nav>
  );
};

export default Navbar;
import Navbar from "./component/navbar/Navbar";
import TokenManager from "./token";
import { useEffect, useState } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./page/Home";
import Photo from "./page/Photo";
import About from "./page/About";
import { UserContext } from "./Context";
import Footer from "./component/footer/footer";
import "./App.css";

function App() {
  const [tokenManager] = useState(() => new TokenManager());
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const init = async () => {
      try {
        if (!tokenManager.hasToken()) {
          await tokenManager.createUser();
        }
      } catch (error) {
        console.error("Error initializing token:", error);
      } finally {
        setLoading(false);
      }
    };
    init();
  }, [tokenManager]);

  if (loading) {
    return <p>Loading app...</p>;
  }

  return (
    <UserContext.Provider value={tokenManager}>
      <BrowserRouter>
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/photo" element={<Photo />} />
          <Route path="/about" element={<About />} />
        </Routes>
        <Footer />
      </BrowserRouter>
    </UserContext.Provider>
  );
}

export default App;
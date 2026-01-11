import { useUser } from "../Context";
import { useEffect, useState } from "react";
import "./../pageCss/Photo.css";

const Photo = () => {
  const tokenManager = useUser();
  const [userData, setUserData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadPhotoData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        console.log("Loading photo data...");
        const data = await tokenManager.loadData();
        console.log("Data loaded:", data);
        setUserData(data);
        
      } catch (err: any) {
        console.error("Error loading photo data:", err);
        setError("Erreur lors du chargement des images");
        
        if (err.status === 404 || err.status === 401) {
          try {
            console.log("Creating new user...");
            await tokenManager.createUser();
            const data = await tokenManager.loadData();
            setUserData(data);
            setError(null);
          } catch (createErr) {
            console.error("Error creating user:", createErr);
          }
        }
      } finally {
        setLoading(false);
      }
    };

    loadPhotoData();
  }, []); 

  if (loading) {
    return (
      <article className="photo-page">
        <p style={{ color: '#fff', fontSize: '1.2rem' }}>Chargement des images...</p>
      </article>
    );
  }

  if (error && !userData) {
    return (
      <article className="photo-page">
        <p style={{ color: '#ff6b6b', fontSize: '1.2rem' }}>{error}</p>
      </article>
    );
  }

  const lowResImages = userData?.userData?.newlow || [];
  const highResImages = userData?.userData?.newhigh || [];

  console.log("Low res images:", lowResImages);
  console.log("High res images:", highResImages);

  return (
    <article className="photo-page">
      <section className="photo-section low-res">
        <header className="photo-header">
          <h2>Low-Resolution Images</h2>
        </header>
        <ul className="photo-grid">
          {lowResImages.length > 0 ? (
            lowResImages.map((image: string, index: number) => (
              <li key={index} className="photo-box">
                <img src={image} alt={`Low-res ${index + 1}`} />
              </li>
            ))
          ) : (
            <li className="photo-box empty">Aucune image disponible</li>
          )}
        </ul>
      </section>

      <section className="photo-section high-res">
        <header className="photo-header">
          <h2>High-Resolution Images</h2>
        </header>
        <ul className="photo-grid">
          {highResImages.length > 0 ? (
            highResImages.map((image: string, index: number) => (
              <li key={index} className="photo-box">
                <img src={image} alt={`High-res ${index + 1}`} />
              </li>
            ))
          ) : (
            <li className="photo-box empty">Aucune image disponible</li>
          )}
        </ul>
      </section>
    </article>
  );
};

export default Photo;
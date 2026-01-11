import React from "react";
import "./uploadSection.css";
import { formDataPost } from "../../fetcher/formData_post";

const apiUrl = import.meta.env.VITE_backPyUrl;

type uplType = "low" | "high";

interface ImageUploadSectionProps {
  title: string;
  description: string;
  type: uplType;
}

const ImageUploadSection: React.FC<ImageUploadSectionProps> = ({
  title,
  description,
  type,
}) => {
  const mutation = formDataPost(`${apiUrl}/img/${type}`);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const originalImgEl = document.getElementById("originalImg") as HTMLImageElement;
    if (originalImgEl) originalImgEl.src = URL.createObjectURL(file);

    const resultsSection = document.querySelector(".results-section");
    if (resultsSection) {
      resultsSection.classList.add("show");
    }

    const formData = new FormData();
    formData.append("image", file);

    mutation.mutate(formData, {
      onSuccess: (data: any) => {
        console.log("Upload success:", data);
        const imgEl = document.getElementById("resultImg") as HTMLImageElement;
        if (imgEl && data.image) {
          if (data.image.startsWith('data:')) {
            imgEl.src = data.image;
          } else {
            imgEl.src = `data:image/webp;base64,${data.image}`;
          }
        }
      },
      onError: (err: any) => {
        console.error("Erreur upload ->", err);
      },
    });
  };

  return (
    <>
      <section className="main">
        <div className="hero">
          <h1>{title}</h1>
          <p>{description}</p>
          
          <label className="inside">
            <input
              type="file"
              accept="image/png,image/jpeg,image/jpg,image/webp,image/avif"
              onChange={handleFileChange}
              disabled={mutation.isPending}
            />
            <span>
              {mutation.isPending 
                ? "Processing..." 
                : "Upload image or drag & drop"}
            </span>
          </label>
        </div>
      </section>

      {/* Nouvelle section pour les résultats */}
      <section className="results-section">
        <div className="results-container">
          <h2 className="results-title">Results</h2>
          
          <div className="images-grid">
            <div className="image-card">
              <span className="image-label">Original Image</span>
              <div className="image-box">
                <img id="originalImg" alt="Image originale" />
              </div>
            </div>

            <div className="image-card">
              <span className="image-label">Result</span>
              <div className="image-box">
                <img id="resultImg" alt="Résultat du modèle" />
                {mutation.isPending && (
                  <div className="loading-overlay">
                    <div className="spinner"></div>
                    <p>Processing...</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </section>
    </>
  );
};

export default ImageUploadSection;
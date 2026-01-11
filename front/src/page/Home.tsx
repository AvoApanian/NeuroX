import ImageUploadSection from "../component/uploadSection/ImageUploadSection";

const Home = () => {
    
  return (
    <main>
      <ImageUploadSection
        title="Low-Resolution Image"
        description="Upload a low-quality image. It will be used as the base input for enhancement."
        type="low"
      />

      <ImageUploadSection
        title="High-Resolution Image"
        description="Upload a high-quality reference image. It will guide the enhancement process."
        type="high"
      />
    </main>
  );
};

export default Home;

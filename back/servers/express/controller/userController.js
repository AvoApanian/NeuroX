const {
  createAccessToken,
  createUserDb,
  loaduserdata
} = require("../service/userServise");
const crypto = require("crypto");

const creationUser = async (req, res) => {
  const userId = crypto.randomUUID();
  // Un seul token avec longue durée (30 jours)
  const token = createAccessToken({ id: userId });
  
  // Stocker le token dans un cookie HTTP-only sécurisé
  res.cookie("userToken", token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production", 
    sameSite: process.env.NODE_ENV === "production" ? "none" : "lax",
    maxAge: 30 * 24 * 60 * 60 * 1000, // 30 jours
  });
  
  await createUserDb(userId);
  
  res.status(200).json({ token });
};

const loadInfo = async (req, res) => {
  const userId = req.user.id;
  
  const userData = await loaduserdata(userId);
  
  if (!userData) {
    return res.status(404).json({ 
      error: "User not found in database" 
    });
  }
  
  res.status(200).json({ userData });
};

module.exports = { 
  creationUser,
  loadInfo
};
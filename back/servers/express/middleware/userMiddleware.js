const jwt = require("jsonwebtoken");

function authMiddleware(req, res, next) {
  const authHeader = req.headers.authorization;

  if (!authHeader) {
    return res.status(401).json(
        { error: "No Authorization header" }
    );
  }

  const [type, token] = authHeader.split(" ");

  if (type !== "Bearer" || !token) {
    return res.status(401).json(
        { error: "Invalid token format" }
    );
  }

  try {
    const decoded = jwt.verify(token, process.env.accesSecret);
    req.user = decoded; 
    next();
  } catch (err) {
    return res.status(401).json(
        { error: "Invalid or expired token" }
    );
  }
}

module.exports = { authMiddleware };

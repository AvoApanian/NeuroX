const jwt = require("jsonwebtoken");
const { Pool } = require("pg");
require("dotenv").config();

const pool = new Pool({
  user: process.env.dbUsersUser,
  password: process.env.dbusersPassword,
  host: process.env.dbUsersHost,
  database: process.env.dbUsersDB,
  port: process.env.dbUsersPort,
});

const tokenSecret = process.env.accesSecret;

function createAccessToken(payload) {
  return jwt.sign(payload, tokenSecret, { expiresIn: "36524d" });
}

async function createUserDb(id) {
  const result = await pool.query(
    "INSERT INTO users (uuid) VALUES ($1) RETURNING *",
    [id]
  );
  return result.rows[0];
}

async function loaduserdata(id) {
  const result = await pool.query(
    "SELECT newhigh, newlow FROM users WHERE uuid = $1",
    [id]
  );
  
  console.log("=== DB QUERY RESULT ===");
  console.log("Raw result:", JSON.stringify(result.rows, null, 2));
  
  if (result.rows.length === 0) {
    console.log(`User ${id} not found in database`);
    return null;
  }
  
  const userData = result.rows[0];
  console.log("Returned userData:", JSON.stringify(userData, null, 2));
  
  return userData;
}

module.exports = {
  pool,
  createAccessToken,
  createUserDb,
  loaduserdata
};
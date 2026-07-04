require("dotenv").config();
const express = require("express");
const http = require("http");
const cors = require("cors");
const jwt = require("jsonwebtoken");
const { Server } = require("socket.io");

const app = express();
app.use(cors());
app.use(express.json());

const JWT_SECRET = process.env.JWT_SECRET;

// Endpoint to generate a room token
// A real app would check the user is logged in etc. — for now, anyone can request one
app.post("/create-room-token", (req, res) => {
  const { roomId } = req.body;
  if (!roomId) return res.status(400).json({ error: "roomId is required" });

  const token = jwt.sign({ roomId }, JWT_SECRET, { expiresIn: "1h" });
  res.json({ token });
});

const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: "*" }
});

const rooms = {};

// Middleware: every socket connection must present a valid token
io.use((socket, next) => {
  const token = socket.handshake.auth.token;
  if (!token) return next(new Error("No token provided"));

  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    socket.roomId = decoded.roomId; // attach the verified room ID to this socket
    next();
  } catch (err) {
    next(new Error("Invalid or expired token"));
  }
});

io.on("connection", (socket) => {
  console.log("New authenticated connection:", socket.id, "for room:", socket.roomId);

  socket.on("join-room", () => {
    const roomId = socket.roomId; // trust the token's room, not client input
    socket.join(roomId);

    if (!rooms[roomId]) rooms[roomId] = [];
    rooms[roomId].push(socket.id);

    console.log(`${socket.id} joined room ${roomId}`);
    socket.to(roomId).emit("user-joined", socket.id);
  });

  socket.on("offer", (payload) => {
    io.to(payload.target).emit("offer", { sdp: payload.sdp, caller: socket.id });
  });

  socket.on("answer", (payload) => {
    io.to(payload.target).emit("answer", { sdp: payload.sdp, caller: socket.id });
  });

  socket.on("ice-candidate", (payload) => {
    io.to(payload.target).emit("ice-candidate", { candidate: payload.candidate, from: socket.id });
  });

  socket.on("disconnect", () => {
    console.log("Disconnected:", socket.id);
    for (const roomId in rooms) {
      rooms[roomId] = rooms[roomId].filter((id) => id !== socket.id);
    }
  });
});

const PORT = 3001;
server.listen(PORT, () => {
  console.log(`Signaling server running on http://localhost:${PORT}`);
});
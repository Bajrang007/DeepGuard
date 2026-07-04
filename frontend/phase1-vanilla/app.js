let socket; // we'll create this only after fetching a token

async function getTokenAndConnect(roomId) {
  const response = await fetch("http://localhost:3001/create-room-token", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ roomId })
  });
  const data = await response.json();

  socket = io("http://localhost:3001", {
    auth: { token: data.token }
  });

  attachSocketListeners();
}

const localVideo = document.getElementById("localVideo");
const remoteVideo = document.getElementById("remoteVideo");
const joinBtn = document.getElementById("joinBtn");
const roomInput = document.getElementById("roomInput");

let localStream;
let peerConnection;
let remoteSocketId;
let currentRoom;

const config = {
  iceServers: [
    { urls: "stun:stun.l.google.com:19302" },
    {
      urls: "turn:localhost:3478",
      username: "testuser",
      credential: "testpassword"
    }
  ]
};

joinBtn.onclick = async () => {
  currentRoom = roomInput.value.trim();
  if (!currentRoom) return alert("Enter a room ID");

  localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
  localVideo.srcObject = localStream;

  await getTokenAndConnect(currentRoom);
  socket.emit("join-room");
};

function attachSocketListeners() {
  socket.on("user-joined", async (otherId) => {
    remoteSocketId = otherId;
    console.log("Another user joined:", otherId);

    peerConnection = createPeerConnection();
    localStream.getTracks().forEach((track) => peerConnection.addTrack(track, localStream));

    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);

    socket.emit("offer", { sdp: offer, target: remoteSocketId });
  });

  socket.on("offer", async ({ sdp, caller }) => {
    remoteSocketId = caller;

    peerConnection = createPeerConnection();
    localStream.getTracks().forEach((track) => peerConnection.addTrack(track, localStream));

    await peerConnection.setRemoteDescription(new RTCSessionDescription(sdp));

    const answer = await peerConnection.createAnswer();
    await peerConnection.setLocalDescription(answer);

    socket.emit("answer", { sdp: answer, target: remoteSocketId });
  });

  socket.on("answer", async ({ sdp }) => {
    await peerConnection.setRemoteDescription(new RTCSessionDescription(sdp));
  });

  socket.on("ice-candidate", async ({ candidate }) => {
    if (candidate) {
      await peerConnection.addIceCandidate(new RTCIceCandidate(candidate));
    }
  });
}

function createPeerConnection() {
  const pc = new RTCPeerConnection(config);

  pc.onicecandidate = (event) => {
    if (event.candidate) {
      socket.emit("ice-candidate", { candidate: event.candidate, target: remoteSocketId });
    }
  };

  pc.ontrack = (event) => {
    remoteVideo.srcObject = event.streams[0];
  };

  return pc;
}
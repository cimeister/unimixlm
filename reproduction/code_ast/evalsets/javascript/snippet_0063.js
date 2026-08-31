const axios = require("axios");
const RM = require("./roomManager");
const moment = require("moment");
const DEFAULT_SPOTIFY_DATA = require("../defaultSpotifyData");
const { v4: uuidv4 } = require("uuid");
const logger = require("../Logger");

class User {
  constructor(
    ip,
    connectedIn,
    spotifyAcssesToken,
    roomManager,
    nickName,
    spotifyID = "notConnected",
    socketID = "notConnected"
  ) {
    this.id = uuidv4();
    this.ip = ip;
    this.connectedIn = connectedIn;
    this.roomID = -1;
    this.roomManager = roomManager;
    this.spotifyID = spotifyID;
    this.nickName = nickName;
    this.socket = "not bined yet";
    this.access_token = spotifyAcssesToken;
    this.favArtists = -1;
    this.chatter = "";
    this.hasActualTaste = false;
  }

  async getSpotifyData() {
    const options1 = {
      headers: { Authorization: "Bearer " + this.access_token },
      json: true,
    };
    const options2 = {
      headers: { Authorization: "Bearer " + this.access_token },
      json: true,
      limit: 50,
    };
    const [userStats, favArtists] = await Promise.all([
      axios.get("https://api.spotify.com/v1/me", options1),
      axios.get("https://api.spotify.com/v1/me/top/artists", options2),
    ]);
    this.spotifyID = userStats.data.id;
    this.favArtists = favArtists.data.items;
    if (this.favArtists.length > 0) {
      this.backgroundImage = this.favArtists[0].images[0];
      this.hasActualTaste = true;
    } else {
      this.favArtists = DEFAULT_SPOTIFY_DATA;
      this.backgroundImage = this.favArtists[0].images[0];
    }
    this.isWaiting = false;
    this.musicTaste = this.GetGeneresFromArtists();
    return true;
  }

  BindToSocket(socket) {
    this.socket = socket;
    if (!this.hasActualTaste) {
      this.socket.emit("noActualTaste");
    }
    this.socket.on("New message", (msg) => {
      logger.info("new message: " + msg);
      this.socket
        .to(this.roomID)
        .emit("New message", msg, moment().format("HH:mm:ss"));
    });
    this.socket.on("typing", () => {
      this.socket.to(this.roomID).emit("typing");
    });
    this.socket.on("disconnect", () => {
      console.log("bye!");
      console.info(this.nickName + " left " + this.chatter.nickName + " alone");

      this.socket.leave(this.roomID);
      this.roomManager.DeleteUser(this);
      if (this.chatter) {
        this.chatter.socket.leave(this.roomID);
        const room = this.roomManager.SearchRoom(this.chatter);
        if (!room) {
          console.log("emitting roomEmpty for " + this.chatter.nickName);
          this.chatter.socket.emit("roomEmpty");
        }
      }
    });
  }
  GetGeneresFromArtists() {
    let geners = [];
    Object.entries(this.favArtists).forEach(([key, artist]) => {
      geners.push(...artist.genres);
    });
    return FindMostCommonGeners(geners, 40);
  }
  FindMatchingGeners(otherUser) {
    let generes = this.musicTaste;
    let otherUserGeners = otherUser.musicTaste;
    return generes.filter((genere) => otherUserGeners.includes(genere));
  }
  JoinRoom(roomName, sharedArtist, sharedGeners) {
    this.socket.join(roomName, () => {
      this.roomID = roomName;
      const { nickName, favArtists } = this.chatter;
      this.socket.emit("enteredRoom", {
        sharedArtist,
        sharedGeners,
        otherUser: { nickName, favArtists },
      });
    });
  }
  ImWaiting() {
    this.isWaiting = true;
    this.chatter = false;
    //  console.log("keep looking you snowflake -" + this.socket.id);
  }
}

function FindMostCommonGeners(generes, howManyGenres) {
  let map = new Map();
  for (let genere of generes) {
    map.set(genere, (map.get(genere) || 0) + 1);
  }
  let mostCommonGeners = [];
  for (let i = 0; i < howManyGenres; i++) {
    let mostCommonGenre = null;
    let maxCount = -1;
    for (let [genere, count] of map.entries()) {
      if (count > maxCount) {
        maxCount = count;
        mostCommonGenre = genere;
      }
    }
    map.delete(mostCommonGenre);
    mostCommonGeners.push(mostCommonGenre);
  }
  return mostCommonGeners;
}

module.exports = User;
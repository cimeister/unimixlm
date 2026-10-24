let vm = new Vue({
    el:'#app',
    data() {
        return {
            isListShow : false,
            isPlayListShow : true,
            isSingerShow : true,
            isAlbumShow : true,
            isFavoriteShow : true,

            isRandom : false,
            isplay : false,
            volumeX : 125, //進度條
            timeX : 0,  //進度條
            currentTime : '0:00', //顯示
            duration : '0:00', //顯示
            whatSong : 1,
            whatlist : 0,
            
            playList : ['工作清單','youtube精選','周杰倫','盧廣仲','魔杰座','依然范特西','12新作','花甲男孩轉大人','幾分之幾'],
            singerList : ['周杰倫','盧廣仲'],
            singerAlbum : [
                ['魔杰座','依然范特西','12新作'],
                ['花甲男孩轉大人','幾分之幾']
            ],
            myFavorite : ['只能想念你','思念是一種病','稻香','幾分之幾','Photograph','End of Summer'],
            
            bigTitle : '工作歌單' ,
            albumImgSrc : 'img/album0.jpg',

            songTitle : '遇見',
            singer : '<NAME>',
            songImgSrc : 'img/album0.jpg',

            workSongs : {
                son1 :{ id:1,title:'遇見',time:"3:32",like:false,singer:'<NAME>',album:'同名專輯',Img:'img/album0.jpg',src:'audio/遇见 Yu Jian （Encounter) - Stefanie Sun 孫燕姿 - Sungha Jung.mp3'},
                son2 :{ id:2,title:'Photograph',time:"3:27",like:true,singer:'<NAME>',album:'同名專輯',Img:'img/album0.jpg',src:'audio/(Ed Sheeran) Photograph - Sungha Jung.mp3'},
                son3 :{ id:3,title:'Some',time:"2:38",like:false,singer:'<NAME>、<NAME>',album:'合作專輯',Img:'img/album00.jpg',src:'audio/썸(Some) - Ahn Jung Jae Sungha Jung.mp3'},
                son4 :{ id:4,title:'End of Summer',time:"1:49",like:true,singer:'<NAME>',album:'無版權音樂2',Img:'img/album10.jpg',src:'audio/End_of_Summer.mp3'},
            },
            youtubeSongs : {
                son1 :{ id:1,title:'<NAME>',time:"3:10",like:false,singer:'<NAME>',album:'無版權音樂',Img:'img/album1.jpg',src:'audio/Lucid_Dreamer.mp3'},
                son2 :{ id:2,title:'Sunspots',time:"5:43",like:false,singer:'<NAME>',album:'無版權音樂2',Img:'img/album10.jpg',src:'audio/Sunspots.mp3'}
            },
            currentAlbum : -1,
            currentList : 0,

            singerSong1 : {
                son1 :{title:'給我一首歌的時間',time:"4:13",like:false,singer:'周杰倫',album:'魔杰座',Img:'img/album2.jpg'},
                son2 :{title:'花海',time:"4:24",like:false,singer:'周杰倫',album:'魔杰座',Img:'img/album2.jpg'},
                son3 :{title:'說好的幸福呢',time:"4:16",like:false,singer:'周杰倫',album:'魔杰座',Img:'img/album2.jpg'},
                son4 :{title:'稻香',time:"3:47",like:true,singer:'周杰倫',album:'魔杰座',Img:'img/album2.jpg'},
                son5 :{title:'夜的第七章',time:"3:48",like:false,singer:'周杰倫',album:'依然范特西',Img:'img/album5.jpg'},
                son6 :{title:'聽媽媽的話',time:"4:25",like:false,singer:'周杰倫',album:'依然范特西',Img:'img/album5.jpg'},
                son7 :{title:'千里之外',time:"4:16",like:false,singer:'周杰倫',album:'依然范特西',Img:'img/album5.jpg'},
                son8 :{title:'紅塵客棧',time:"4:34",like:false,singer:'周杰倫',album:'12新作',Img:'img/album6.jpg'},
                son9 :{title:'明明就',time:"4:18",like:false,singer:'周杰倫',album:'12新作',Img:'img/album6.jpg'},
                son10 :{title:'比較大的大提琴',time:"4:13",like:false,singer:'周杰倫、梁心頤、楊瑞代',album:'12新作',Img:'img/album6.jpg'},                
            },
            singerSong2 : {
                son1 :{title:'明仔載-演奏版',time:"4:34",like:false,singer:'盧廣仲',album:'花甲男孩轉大人',Img:'img/album3.jpg'},
                son2 :{title:'繁星五號',time:"4:14",like:false,singer:'盧廣仲',album:'花甲男孩轉大人',Img:'img/album3.jpg'},
                son3 :{title:'快魚仔',time:"3:12",like:false,singer:'盧廣仲',album:'花甲男孩轉大人',Img:'img/album3.jpg'},
                son4 :{title:'幾分之幾',time:"4:54",like:true,singer:'盧廣仲',album:'幾分之幾',Img:'img/album8.jpg'},
                son5 :{title:'你是我的水',time:"3:30",like:false,singer:'盧廣仲',album:'幾分之幾',Img:'img/album8.jpg'},
            }
        }
    },
    computed:{
        outPutIcon(){
            if(this.isplay){
                return 'url(img/stop.svg)'
            }else{
                return 'url(img/Polygon_9.svg)'
            }
        },
        colorV(){
            let percent = this.volumeX / 152 * 100
            return 'linear-gradient(to right, #c5c5c5 '+percent+'% , #fff '+percent+'%)'
        },
        colorT(){
            let percent = this.timeX / 552 * 100
            return 'linear-gradient(to right, #c5c5c5 '+percent+'% , #fff '+percent+'%)'
        },
        timeqqq(){
            let audio = document.querySelector('#audio');
            let xxx = audio.currentTime / audio.duration * 552
            this.timeX = xxx
        },
        AlbumSongs1(){
            let test = new Object()
            test.son1 = this.singerSong1.son1
            test.son2 = this.singerSong1.son2
            test.son3 = this.singerSong1.son3
            test.son4 = this.singerSong1.son4
            return test
        },
        AlbumSongs2(){
            let test = new Object()
            test.son5 = this.singerSong1.son5
            test.son6 = this.singerSong1.son6
            test.son7 = this.singerSong1.son7
            return test
        },
        AlbumSongs3(){
            let test = new Object()
            test.son8 = this.singerSong1.son8
            test.son9 = this.singerSong1.son9
            test.son10 = this.singerSong1.son10
            return test
        },
        AlbumSongs4(){
            let test = new Object()
            test.son1 = this.singerSong2.son1
            test.son2 = this.singerSong2.son2
            test.son3 = this.singerSong2.son3
            return test
        },
        AlbumSongs5(){
            let test = new Object()
            test.son4 = this.singerSong2.son4
            test.son5 = this.singerSong2.son5
            return test
        },
    },
    methods:{
        listShow(){
            this.isListShow = !this.isListShow;
        },
        mousedownVol(e){
            let or = this.volumeX;
            let audio = document.querySelector('#audio');
            document.onmousemove = (e2) => {
                let Xer = (e2.clientX-e.clientX);
                if(Xer + or  > 152){
                    this.volumeX == 152
                }else if(Xer + or < 0){
                    this.volumeX == 0
                }else{
                    this.volumeX = Xer + or
                }
                audio.volume = this.volumeX / 152 ;
            };
            document.onmouseup = ()=> {
                document.onmousemove=null;
                document.onmousedown=null;
            };
        },
        mousedownTime(e){
            let or = this.timeX
            let audio = document.querySelector('#audio');
            document.onmousemove = (e2) => {
                let Xer = (e2.clientX-e.clientX);
                if(Xer + or  > 552){
                    this.timeX == 552
                }else if(Xer + or < 0){
                    this.timeX == 0
                }else{
                    this.timeX = Xer + or
                }
                audio.currentTime = this.timeX / 552 * audio.duration;
            };
            document.onmouseup = ()=> {
                document.onmousemove=null;
                document.onmousedown=null;
            };
        },
        playOrStop(){
            let audio = document.querySelector('#audio');
            if(audio.src.indexOf('audio')<0)return
            if(!this.isplay){
                audio.play();
                this.isplay = true;
            }else if(this.isplay){
                audio.pause();
                this.isplay = false;
            }
        },
        minSec(time){
            let min = Math.floor(time/60);
            let sec = '0' + Math.round(time%60);
            let sec2 = sec.slice(-2)
            return min+':'+sec2;
        },
        timeUpdate(){
            let timeStr = parseInt(this.$refs.mp3.currentTime);
            this.currentTime = this.minSec(timeStr);
            this.timeX = timeStr / this.$refs.mp3.duration * 552
        },
        canPlay(){
            this.duration = this.minSec(this.$refs.mp3.duration);
        },
        end(){
            this.isplay = false
            this.timeX = 0
            let audio =document.querySelector('#audio');
            audio.currentTime = 0
        },
        selectSong(event){
            let Src = event.currentTarget.getAttribute("num")
            let title = event.currentTarget.getAttribute('til')
            let sir = event.currentTarget.getAttribute('sir')
            let img = event.currentTarget.getAttribute('img')
            let ind = event.currentTarget.getAttribute('ind')
            let audio = document.querySelector("#audio")
            this.end()
            audio.src = Src
            this.songTitle = title
            this.singer = sir
            this.songImgSrc = img
            this.whatSong = ind
            this.whatlist = this.currentList
            this.playOrStop()
        },
        selectAd(event){
            let title = event.currentTarget.getAttribute('til')
            let sir = event.currentTarget.getAttribute('sir')
            let img = event.currentTarget.getAttribute('img')
            let audio = document.querySelector("#audio")
            this.end()
            audio.src = ''
            this.songTitle = title
            this.singer = sir
            this.songImgSrc = img
            let ad = document.querySelector('.ad')
            ad.classList.add('showAd')
        },
        targetList(event){
            let el = document.querySelector(".point_active")
            el.classList.remove('point_active')
            let el2 = event.currentTarget
            el2.classList.add('point_active')
        },
        changeAlbum(index){
            this.currentAlbum = index
        },
        changeList(index){
            this.currentList = index
            this.albumImgSrc = `img/album${index}.jpg`
            this.bigTitle = this.playList[index]
        },
        likeHeart(){
            let ppp = event.currentTarget.getAttribute('num')
            if(this.currentList==0){
                this.workSongs[ppp].like = !this.workSongs[ppp].like
                if(this.workSongs[ppp].like){
                    this.myFavorite.push(this.workSongs[ppp].title)
                }else{
                    let num = this.myFavorite.indexOf(this.workSongs[ppp].title)
                    this.myFavorite.splice(num, 1)
                }
            }
            else if(this.currentList==1){
                this.youtubeSongs[ppp].like = !this.youtubeSongs[ppp].like
                if(this.youtubeSongs[ppp].like){
                    this.myFavorite.push(this.youtubeSongs[ppp].title)
                }else{
                    let num = this.myFavorite.indexOf(this.youtubeSongs[ppp].title)
                    this.myFavorite.splice(num, 1)
                }
            }else if(this.currentList==2||this.currentList==4||this.currentList==5||
                this.currentList==6){
                this.singerSong1[ppp].like = !this.singerSong1[ppp].like
                if(this.singerSong1[ppp].like){
                    this.myFavorite.push(this.singerSong1[ppp].title)
                }else{
                    let num = this.myFavorite.indexOf(this.singerSong1[ppp].title)
                    this.myFavorite.splice(num, 1)
                }
            }else if (this.currentList==3||this.currentList==7||this.currentList==8){
                this.singerSong2[ppp].like = !this.singerSong2[ppp].like
                if(this.singerSong2[ppp].like){
                    this.myFavorite.push(this.singerSong2[ppp].title)
                }else{
                    let num = this.myFavorite.indexOf(this.singerSong2[ppp].title)
                    this.myFavorite.splice(num, 1)
                }
            }
        },
        closeAd(){
            let ad = document.querySelector('.ad')
            ad.classList.remove('showAd')
        },
        switchMode(e){
            let btn = document.querySelector('.recycle')
            if(this.isRandom){
                btn.classList.remove('active')
            }else if(!this.isRandom){
                btn.classList.add('active')
            }
            this.isRandom = !this.isRandom
        },
        preSong(){
            this.timeX = 0
            let audio =document.querySelector('#audio');
            audio.currentTime = 0
            let num = parseInt(this.whatSong) -1
            let list = this.whatlist
            if(list==0){
                if(num<1){num=4}
                audio.src = this.workSongs["son"+num].src
                this.songTitle = this.workSongs["son"+num].title
                this.singer = this.workSongs["son"+num].singer
                this.songImgSrc = this.workSongs["son"+num].Img
                this.whatSong = num
            }else if(list==1){
                if(num<1){num=2}
                audio.src = this.youtubeSongs["son"+num].src
                this.songTitle = this.youtubeSongs["son"+num].title
                this.singer = this.youtubeSongs["son"+num].singer
                this.songImgSrc = this.youtubeSongs["son"+num].Img
                this.whatSong = num
            }
            if(!this.isplay){this.playOrStop()}
            else{audio.play()}
        },
        nextSong(){
            this.timeX = 0
            let audio =document.querySelector('#audio');
            audio.currentTime = 0
            let num = parseInt(this.whatSong) + 1
            let list = this.whatlist
            if(list==0){
                if(num>4){num=1}
                audio.src = this.workSongs["son"+num].src
                this.songTitle = this.workSongs["son"+num].title
                this.singer = this.workSongs["son"+num].singer
                this.songImgSrc = this.workSongs["son"+num].Img
                this.whatSong = num
            }else if(list==1){
                if(num>2){num=1}
                audio.src = this.youtubeSongs["son"+num].src
                this.songTitle = this.youtubeSongs["son"+num].title
                this.singer = this.youtubeSongs["son"+num].singer
                this.songImgSrc = this.youtubeSongs["son"+num].Img
                this.whatSong = num
            }
            if(!this.isplay){this.playOrStop()}
            else{audio.play()}
        },
        randomSong(){
            if(this.isRandom){
                this.timeX = 0
                let audio =document.querySelector('#audio');
                audio.currentTime = 0
                let list = this.whatlist
                let randomNum = Math.floor(Math.random()*100);
                if(list==0){
                    let num = randomNum%4
                    console.log(num)
                    if(num==0){num=4}
                    if(this.whatSong==num){if(this.whatSong==1){num=4}else{num-=1}}
                    audio.src = this.workSongs["son"+num].src
                    this.songTitle = this.workSongs["son"+num].title
                    this.singer = this.workSongs["son"+num].singer
                    this.songImgSrc = this.workSongs["son"+num].Img
                    this.whatSong = num
                    audio.play()
                }else if(list==1){
                    this.nextSong() //因為第二個歌單只有兩首 所以算了ＸＤ
                }
            }else{
                this.nextSong()
            }
        }
    }
})
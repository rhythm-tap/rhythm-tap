

// 要素取得
let description_top = document.getElementById('descriptionTop');
let description_bottom = document.getElementById('descriptionBottom');
let input_name = document.getElementById('inputName');
let btn1 = document.getElementById('btn1');
let btn2 = document.getElementById('btn2');
let countdown = document.getElementById('countdown');

// 音楽
let audio = new Audio('/static/audio/merry.mp3');
const music_start_position = 0;
let save_muted = true;

// 変数定義
let player_name, start_time, finish_time, data = [];
let ax, ay, az, alpha, beta, gamma;

// データ取得の周期(ms)
const data_period = 10;

// タップ終了ボタン有無
const fin_button_exist = true;

// 変数初期化
initVariable();

// 最初のページ
window.addEventListener('load', phaseInputName);

// 名前入力フェーズ[1]
function phaseInputName(){
    description_top.innerHTML = "名前を入力してください";
    description_bottom.style.display = 'none';
    input_name.style.display = 'block';
    // input_name.focus();
    btn1.style.display = 'none';
    btn1.querySelector('button').classList.remove('btn-green');
    btn1.querySelector('button').classList.add('btn-blue');
    btn2.querySelector('button').innerText = "パターン登録へ";
    btn2.querySelector('button').onclick = function(){
        if(input_name.value===''){
            alert("名前が未入力です");
            input_name.focus();
            return;
        }
        vibration([200]);
        player_name = String(input_name.value);
        phaseListenMusic();
    };
    btn2.querySelector('button').classList.remove('btn-gray');
    btn2.querySelector('button').classList.add('btn-green');
    countdown.style.display = 'none';
    // if( !window.DeviceMotionEvent || !window.DeviceOrientationEvent || (getOperatingSystem()!=='Android') ){
    //     alert("このデバイスはタップ認証に対応していません。");
    //     window.location.href = "/";
    // }
}

// 音楽を聴くフェーズ[2]
function phaseListenMusic(){
    description_top.innerHTML = "「メリーさんの羊」が流れます。リズムを確認してください。";
    description_bottom.style.display = 'none';
    input_name.style.display = 'none';
    btn1.style.display = 'flex';
    btn1.querySelector('button').classList.remove('btn-green');
    btn1.querySelector('button').classList.add('btn-blue');
    btn1.querySelector('button').innerText = "再生";
    btn1.querySelector('button').onclick = playMusic;
    btn2.style.display = 'flex';
    btn2.querySelector('button').classList.remove('btn-gray');
    btn2.querySelector('button').classList.add('btn-green');
    btn2.querySelector('button').innerText = "リズムパターンの登録";
    btn2.querySelector('button').onclick = function(){
        stopMusic();
        phaseStartRegist();
    };
    countdown.style.display = 'none';
}

// 登録開始フェーズ[3]
function phaseStartRegist(){
    vibration([200,200,200]);
    description_top.innerHTML = "リズムパターンの登録を開始します。<br>準備ができたら開始ボタンを押してください。";
    description_bottom.style.display = 'block';
    description_bottom.innerHTML = "カウントダウン後にリズムパターンをタップしてください。<br>※ 画面・側面・背面のどの面をタップしても構いません。";
    input_name.style.display = 'none';
    if(save_muted){
        muteMusic();
    }else{
        unmuteMusic();
    }
    btn1.querySelector('button').onclick = toggleMute;
    btn2.style.display = 'flex';
    btn2.querySelector('button').innerText = "開始";
    btn2.querySelector('button').onclick = function(){
        phaseCountDown();
    };
    countdown.style.display = 'none';
}

// カウントダウンフェーズ[4]
let interval_id = null;
let save_data_interval_id = null;
function phaseCountDown(){
    countdown.style.display = 'flex';
    countdown.innerText = 3;
    vibration([100]);
    interval_id = setInterval(function(){
        let num = parseInt(countdown.innerText) - 1;
        if(num > 0){
            vibration([100]);
            countdown.innerText = num;
        }else{
            window.setTimeout(playMusic, 1000);
            phaseRegist();
        }
        if(num === 1){
            start_time = Date.now();
            saveData("s");
            save_data_interval_id = setInterval(saveData, data_period);
        }
    }, 1000);
}

// 登録フェーズ[5]
function phaseRegist(){
    clearInterval(interval_id);
    description_top.innerHTML = "リズムパターンをタップしてください";
    description_bottom.style.display = 'none';
    input_name.style.display = 'none';
    btn1.style.display = 'none';
    btn2.style.display = 'flex';
    btn2.querySelector('button').innerText = "タップ終了";
    if(!fin_button_exist){
        btn2.style.display = "none";
    }
    btn2.querySelector('button').onclick = function(){
        if(window.timeout_id !== undefined){
            clearTimeout(window.timeout_id);
            window.timeout_id = undefined;
        }
        stopMusic();
        clearInterval(save_data_interval_id);
        saveData("f");
        finish_time = Date.now();
        sendData(data);
        phasefinishRegist();
    };
    // 15秒後にはタップ終了にする
    window.timeout_id = setTimeout(function(){
        btn2.querySelector('button').click();
    }, 15000);
    countdown.style.display = 'none';
}

// 登録終了フェーズ[6]
function phasefinishRegist(){
    vibration([100,200,100,200,100]);
    description_top.innerHTML = "リズムパターンの登録が完了しました";
    description_bottom.style.display = 'none';
    input_name.style.display = 'none';
    btn1.style.display = 'flex';
    btn1.querySelector('button').classList.remove('btn-blue');
    btn1.querySelector('button').classList.add('btn-green');
    btn1.querySelector('button').innerText = "もう一度計測する";
    btn1.querySelector('button').onclick = function(){
        initVariable();
        phaseListenMusic();
    };
    btn2.style.display = 'flex';
    btn2.querySelector('button').innerText = "アプリ終了";
    btn2.querySelector('button').classList.add('btn-gray');
    btn2.querySelector('button').classList.remove('btn-green');
    btn2.querySelector('button').onclick = function(){
        window.location.href = "/";
    };
    countdown.style.display = 'none';
}





// 変数初期化
function initVariable(){
    // 音楽
    audio.loop = false;
    audio.autoplay = false;
    audio.muted = false;
    audio.controls = false;
    audio.preload = 'metadata';
    audio.currentTime = music_start_position;
    // 変数
    // player_name = undefined;
    start_time = undefined;
    finish_time = undefined;
    data = [];
    ax = undefined;
    ay = undefined;
    az = undefined;
    alpha = undefined;
    beta = undefined;
    gamma = undefined;
}

// データ送信
function sendData(regist_data){
	const xhr = new XMLHttpRequest();
	const fd = {};
	xhr.open('post', '/post_regist_data');
    xhr.setRequestHeader("Content-Type", "application/json");
    fd.name = player_name;
    fd.start_time = start_time;
    fd.finish_time = finish_time;
    fd.regist_data = regist_data;
	xhr.addEventListener('readystatechange', () => {
		if( xhr.readyState === 4 && xhr.status === 200) {
			console.log(xhr.response);
		}
		if( xhr.readyState === 4 && xhr.status === 400) {
			console.log(xhr.response);
		}
    });
	xhr.send(JSON.stringify(fd));
}



// 加速度取得
function getAcceleration(event){
    if(getOperatingSystem() === "iOS"){
        ax = -event.acceleration.x;
        ay = -event.acceleration.y;
        az = -event.acceleration.z;
    }else{
        ax = event.acceleration.x;
        ay = event.acceleration.y;
        az = event.acceleration.z;
    }
}
window.addEventListener("devicemotion", getAcceleration);
// ジャイロ取得
function getGyro(event){
    alpha = event.alpha;
    beta = event.beta;
    gamma = event.gamma;
}
window.addEventListener("deviceorientation", getGyro);
// データ保存(リストに追加)
function saveData(flag=""){
    let timestamp = Date.now();
    data.push(
        [timestamp,ax,ay,az,alpha,beta,gamma,flag]
    );
}






// デバイスのOS検出
function getOperatingSystem() {
    var userAgent = navigator.userAgent || navigator.vendor || window.opera;
    if (/windows phone/i.test(userAgent)) return 'Windows Phone';
    if (/win/i.test(userAgent)) return 'Windows';
    if (/iPad|iPhone|iPod/.test(userAgent) && !window.MSStream) return 'iOS';
    if (/android/i.test(userAgent)) return 'Android';
    if (/mac/i.test(userAgent)) return 'macOS';
    if (/Linux/.test(userAgent)) return 'Linux';
    return 'unknown';
}



// 音楽系の処理
async function playMusic(){
    try {
        await audio.play();
        btn1.querySelector('button').innerText = "停止";
        btn1.querySelector('button').classList.add('btn-red');
        btn1.querySelector('button').classList.remove('btn-blue');
        btn1.querySelector('button').onclick = stopMusic;
    }catch (err) {
        console.warn(err)
    }
}
function stopMusic(){
    if (audio.paused) {
        audio.currentTime = music_start_position;
    }else{
        audio.pause();
        audio.currentTime = music_start_position;
    }
    btn1.querySelector('button').innerText = "もう一度再生";
    btn1.querySelector('button').classList.remove('btn-red');
    btn1.querySelector('button').classList.add('btn-blue');
    btn1.querySelector('button').onclick = playMusic;
}
function toggleMute(){
    if(btn1.querySelector('button').innerText === "ミュートにする"){
        muteMusic();
    }else{
        unmuteMusic();
    }
}
function muteMusic(){
    audio.muted = true;
    save_muted = true;
    btn1.querySelector('button').innerText = "ミュート解除";
    btn1.querySelector('button').classList.add('btn-red');
    btn1.querySelector('button').classList.remove('btn-blue');
}
function unmuteMusic(){
    audio.muted = false;
    save_muted = false;
    btn1.querySelector('button').innerText = "ミュートにする";
    btn1.querySelector('button').classList.remove('btn-red');
    btn1.querySelector('button').classList.add('btn-blue');
}
audio.addEventListener('ended', stopMusic);


// バイブレーション
function vibration(ms_list=[100]){
    if(navigator.vibrate){
        navigator.vibrate(ms_list);
    }else if(navigator.mozVibrate){
        navigator.mozVibrate(ms_list);
    }else if(navigator.webkitVibrate){
        navigator.webkitVibrate(ms_list);
    }else{
        console.log("Not support.");
    }
}

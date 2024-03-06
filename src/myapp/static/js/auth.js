

// 要素取得
let description_top = document.getElementById('descriptionTop');
let description_bottom = document.getElementById('descriptionBottom');
let input_name = document.getElementById('inputName');
let btn1 = document.getElementById('btn1');
let btn2 = document.getElementById('btn2');
let countdown = document.getElementById('countdown');

// 認証用音楽
let correct_audio = new Audio('/static/audio/correct.mp3');
correct_audio.preload = 'auto';
correct_audio.volume = 0.5;
let wrong_audio = new Audio('/static/audio/wrong.mp3');
wrong_audio.preload = 'auto';
wrong_audio.volume = 0.5;

// 変数定義
let player_name, start_time, finish_time, data = [];
let ax, ay, az, alpha, beta, gamma;

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
    input_name.focus();
    btn1.querySelector('button').innerText = "パターン認証へ";
    btn1.querySelector('button').onclick = function(){
        if(input_name.value===''){
            alert("名前が未入力です");
            input_name.focus();
            return;
        }
        vibration([200]);
        player_name = String(input_name.value);
        phaseStartRegist();
    };
    btn2.style.display = 'none';
    btn2.querySelector('button').classList.remove('btn-gray');
    btn2.querySelector('button').classList.add('btn-green');
    countdown.style.display = 'none';
    // if( !window.DeviceMotionEvent || !window.DeviceOrientationEvent || (getOperatingSystem()!=='Android') ){
    //     alert("このデバイスはタップ認証に対応していません。");
    //     window.location.href = "/";
    // }
}


// 登録開始フェーズ[3]
function phaseStartRegist(){
    vibration([200,200,200]);
    description_top.innerHTML = "リズムパターンの登録を開始します。<br>準備ができたら開始ボタンを押してください。";
    description_bottom.style.display = 'block';
    description_bottom.innerHTML = "カウントダウン後にリズムパターンをタップしてください。<br>※ 画面・側面・背面のどの面をタップしても構いません。";
    input_name.style.display = 'none';
    btn1.style.display = 'none';
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
            phaseRegist();
        }
        if(num === 1){
            start_time = Date.now();
            saveData("s");
            save_data_interval_id = setInterval(saveData, 10);
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
    // description_top.innerHTML = "認証中です...";
    description_bottom.style.display = 'none';
    input_name.style.display = 'none';
    btn1.style.display = 'flex';
    btn1.querySelector('button').innerText = "もう一度認証する";
    btn1.querySelector('button').onclick = function(){
        initVariable();
        phaseInputName();
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
    // 変数
    // player_name = undefined;
    start_time = undefined;
    finish_time = undefined;
    data = [];
    // ax = undefined;
    // ay = undefined;
    // az = undefined;
    // alpha = undefined;
    // beta = undefined;
    // gamma = undefined;
    ax = 1;
    ay = 2;
    az = 3;
    alpha = 4;
    beta = 5;
    gamma = 6;
}

// データ送信
function sendData(auth_data){
	const xhr = new XMLHttpRequest();
	const fd = {};
	xhr.open('post', '/post_auth_data');
    xhr.setRequestHeader("Content-Type", "application/json");
    fd.name = player_name;
    fd.start_time = start_time;
    fd.finish_time = finish_time;
    fd.auth_data = auth_data;
	xhr.addEventListener('readystatechange', () => {
		if( xhr.readyState === 4 && xhr.status === 200 ) {
            let res = JSON.parse(xhr.response);
            console.log(res);
            description_bottom.style.display = 'block';
            let res_detail = "";
            for(let i=0; i<5; i++){
                if(i in res.auth_result){
                    if(res_detail === ""){
                        res_detail += String(res.auth_result[i])
                    }else{
                        res_detail += ', '+String(res.auth_result[i])
                    }
                }
            }
            description_bottom.innerHTML = res_detail;
            if(res.auth_result.result){
                description_top.innerHTML = "認証が完了しました。"+String(res.auth_result.auth_name)+"さんこんにちは。";
                correct_audio.play();
            }else{
                description_top.innerHTML = "認証に失敗しました。"+String(res.auth_result.auth_name)+"さんではないと判断されました。";
                wrong_audio.play();
            }
		}
		if( xhr.readyState === 4 && xhr.status === 400 ) {
            let res = JSON.parse(xhr.response);
            console.log(res);
            description_top.innerHTML = xhr.response;
		}
        if( xhr.readyState === 4 && xhr.status === 500 ) {
            let res = JSON.parse(xhr.response);
            console.log(res);
            description_top.innerHTML = xhr.response;
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

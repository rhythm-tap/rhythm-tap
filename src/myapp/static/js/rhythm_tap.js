



let chartColors = {
    red: 'rgb(255, 99, 132)',
    orange: 'rgb(255, 159, 64)',
    yellow: 'rgb(255, 205, 86)',
    green: 'rgb(75, 192, 192)',
    blue: 'rgb(54, 162, 235)',
    purple: 'rgb(153, 102, 255)',
    grey: 'rgb(201, 203, 207)'
};


// 加速度センサー
if(location.pathname.match(/demo\/acceleration/g)){
    if(window.DeviceMotionEvent){
          
        var ctx = document.getElementById('myChart').getContext('2d');
        var chart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'x',
                    borderColor: chartColors.red,
                    fill: false,
                    pointRadius: 0,
                    data: []
                },
                {
                    label: 'y',
                    borderColor: chartColors.blue,
                    fill: false,
                    pointRadius: 0,
                    data: []
                },
                {
                    label: 'z',
                    borderColor: chartColors.yellow,
                    fill: false,
                    pointRadius: 0,
                    data: []
                }]
            },
            options: {
                legend: {
                    position: 'bottom'
                },
                scales: {
                    yAxes: [{
                        display: true,
                        ticks: {
                            min: -3,
                            max: 3,
                        }
                    }],
                    xAxes: [{
                        type: 'realtime',
                        realtime: {
                            delay: 30,
                            onRefresh: function(chart) {
                                chart.data.datasets[0].data.push({
                                    x: Date.now(),
                                    y: acceleration[0]
                                });
                                chart.data.datasets[1].data.push({
                                    x: Date.now(),
                                    y: acceleration[1]
                                });
                                chart.data.datasets[2].data.push({
                                    x: Date.now(),
                                    y: acceleration[2]
                                });
                            }
                        }
                    }]
                }
            }
        });

        let acceleration = [];
        let lastFire = Date.now();
        let count = 0;
        let max_frequency = -1;
        const output_freq = 10;

        document.getElementById("result").innerHTML = "DeviceMotionEventに対応しています。";
        window.addEventListener("devicemotion", function(event){
            document.getElementById("result").innerHTML = "加速度センサーの変化を検知しました。";

            let currentFire = Date.now();
            let period = currentFire - lastFire; // 周期を計算します
            lastFire = currentFire; // 前回の発火時間を更新します
            let frequency = 1000 / period;
            if(frequency > max_frequency){
                max_frequency = frequency;
            }
            count++;

            const ax = event.acceleration.x;    // x軸の重力加速度（Android と iOSでは正負が逆）
            const ay = event.acceleration.y;    // y軸の重力加速度（Android と iOSでは正負が逆）
            const az = event.acceleration.z;    // z軸の重力加速度（Android と iOSでは正負が逆）
            acceleration = [ax, ay, az];

            let freq_elem = document.getElementById('frequency');
            let max_freq_elem = document.getElementById('max_frequency');
            let ax_elem = document.getElementById('ax');
            let ay_elem = document.getElementById('ay');
            let az_elem = document.getElementById('az');
            if(ax_elem && ay_elem && az_elem && (count%output_freq===0)){
                freq_elem.innerHTML = "周波数: "+String(Math.round(frequency))+" Hz";
                max_freq_elem.innerHTML = "最大周波数: "+String(Math.round(max_frequency))+" Hz";
                ax_elem.innerHTML = ax;
                ay_elem.innerHTML = ay;
                az_elem.innerHTML = az;
            }else{
                console.log('x: '+String(ax));
                console.log('y: '+String(ay));
                console.log('z: '+String(az));
            }
        }, true);
    }else{
        document.getElementById("result").innerHTML = "DeviceMotionEventに対応していません。";
    }
}


// ジャイロセンサー
if(location.pathname.match(/demo\/gyro/g)){
    if (window.DeviceOrientationEvent) {
          
        var ctx = document.getElementById('myChart').getContext('2d');
        var chart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'x',
                    borderColor: chartColors.red,
                    fill: false,
                    pointRadius: 0,
                    data: []
                },
                {
                    label: 'y',
                    borderColor: chartColors.blue,
                    fill: false,
                    pointRadius: 0,
                    data: []
                },
                {
                    label: 'z',
                    borderColor: chartColors.yellow,
                    fill: false,
                    pointRadius: 0,
                    data: []
                }]
            },
            options: {
                legend: {
                    position: 'bottom'
                },
                scales: {
                    xAxes: [{
                    type: 'realtime',
                    realtime: {
                        delay: 30,
                        onRefresh: function(chart) {
                            chart.data.datasets[0].data.push({
                                x: Date.now(),
                                y: gyro[0]
                            });
                            chart.data.datasets[1].data.push({
                                x: Date.now(),
                                y: gyro[1]
                            });
                            chart.data.datasets[2].data.push({
                                x: Date.now(),
                                y: gyro[2]
                            });
                        }
                    }
                    }]
                }
            }
        });

        let gyro = [];
        let lastFire = Date.now();
        let count = 0;
        let max_frequency = -1;
        const output_freq = 10;

        document.getElementById("result").innerHTML = "DeviceOrientationEventに対応しています。";
        window.addEventListener('deviceorientation', function(event) {
            document.getElementById("result").innerHTML = "ジャイロセンサーの変化を検知しました。";
            
            let currentFire = Date.now();
            let period = currentFire - lastFire; // 周期を計算します
            lastFire = currentFire; // 前回の発火時間を更新します
            let frequency = 1000 / period;
            if(frequency > max_frequency){
                max_frequency = frequency;
            }
            count++;

            const alpha = event.alpha;
            const beta = event.beta;
            const gamma = event.gamma;
            gyro = [alpha, beta, gamma];

            let freq_elem = document.getElementById('frequency');
            let max_freq_elem = document.getElementById('max_frequency');
            let alpha_elem = document.getElementById('alpha');
            let beta_elem = document.getElementById('beta');
            let gamma_elem = document.getElementById('gamma');
            if(alpha_elem && beta_elem && gamma_elem && (count%output_freq===0)){
                freq_elem.innerHTML = "周波数: "+String(Math.round(frequency))+" Hz";
                max_freq_elem.innerHTML = "最大周波数: "+String(Math.round(max_frequency))+" Hz";
                alpha_elem.innerHTML = alpha;
                beta_elem.innerHTML = beta;
                gamma_elem.innerHTML = gamma;
            }else{
                console.log('alpha: '+String(alpha));
                console.log('beta: '+String(beta));
                console.log('gamma: '+String(gamma));
            }
        });
    } else {
        document.getElementById("result").innerHTML = "DeviceOrientationEventに対応していません。";
    }
}







// バイブレーション
if(location.pathname.match(/demo\/vibration/g) !== null){
    document.getElementById('btn').addEventListener('click', function(){
        if(navigator.vibrate){
            navigator.vibrate(2000);
        }else if(navigator.mozVibrate){
            navigator.mozVibrate(2000);
        }else if(navigator.webkitVibrate){
            navigator.webkitVibrate(2000);
        }else{
            alert("Not support.");
        }
    });
}





// 音楽再生
if(location.pathname.match(/demo\/audio_output/g) !== null){
    document.getElementById('btn').addEventListener('click', function(){
        var audio = new Audio('/assets/audio/merry.mp3#t=0,11');
        audio.play();
    });
}



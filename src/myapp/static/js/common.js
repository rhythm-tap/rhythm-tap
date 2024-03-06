

////////////////////////////////////////////////////////////////////////////////////////////////////
// ハンバーガー
////////////////////////////////////////////////////////////////////////////////////////////////////

window.addEventListener('load', function(){
    if(document.getElementById('humberger_button') && document.querySelector('.humberger-menu')){
        document.getElementById('humberger_button').addEventListener('click', switchHumberger);
        document.querySelector('.humberger-menu').addEventListener('click', switchHumberger);
    }
});
function switchHumberger(){
    if(document.getElementById('humberger_button').getAttribute('aria-expanded')==='false'){
        // ハンバーガーマーク
        document.getElementById('humberger_button').classList.add('humberger-active');
        // ハンバーガーメニュー
        document.querySelector('.humberger-menu').classList.add('humberger-active');
        // スクロール禁止
        window.past_overflow = document.body.style.overflow;
        document.body.style.overflow = 'hidden';
        // 状態切り替え
        document.getElementById('humberger_button').setAttribute('aria-expanded','true');
    }else{
        // ハンバーガーマーク
        document.getElementById('humberger_button').classList.remove('humberger-active');
        // ハンバーガーメニュー
        document.querySelector('.humberger-menu').classList.remove('humberger-active');
        // スクロール解除
        document.body.style.overflow = window.past_overflow;
        // 状態切り替え
        document.getElementById('humberger_button').setAttribute('aria-expanded','false');
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////////////////////////
// ローディング
////////////////////////////////////////////////////////////////////////////////////////////////////

window.addEventListener("load", endLoading);
setTimeout(endLoading, 10*1000);
function endLoading(){
    var elem = document.getElementById('loading');
    if(elem!==null){
        elem.classList.add('loaded');
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////




////////////////////////////////////////////////////////////////////////////////////////////////////
// ヘッダーユーザーアイコンを押下時のドロップダウンメニュー
////////////////////////////////////////////////////////////////////////////////////////////////////

window.addEventListener('load',function(){
    let icon_elem = document.querySelector('header .login-status img');
    if(icon_elem){
        icon_elem.addEventListener('click',function(){
            let dropdown_elem = document.querySelector('header .login-status .dropdown');
            if(dropdown_elem){
                if(dropdown_elem.style.display==='none'){
                    dropdown_elem.style.display = 'block';
                }else if(dropdown_elem.style.display==='block'){
                    dropdown_elem.style.display = 'none';
                }else{
                    dropdown_elem.style.display = 'block';
                }
            }
        });
    }
});


////////////////////////////////////////////////////////////////////////////////////////////////////






////////////////////////////////////////////////////////////////////////////////////////////////////
// 確認画面からの戻るボタン押下時
////////////////////////////////////////////////////////////////////////////////////////////////////

function setModeBack(){
    var elem = document.querySelector('input[name="mode"]');
    if(elem!==null){
        elem.value = "back";
        return true;
    }else{
        return false;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////








////////////////////////////////////////////////////////////////////////////////////////////////////
// リンクをコピー
////////////////////////////////////////////////////////////////////////////////////////////////////

function copyUrl() {
    const element = document.createElement('input');
    element.value = location.href;
    document.body.appendChild(element);
    element.select();
    document.execCommand('copy');
    document.body.removeChild(element);
    alert("リンクをクリップボードにコピーしました。");
}

////////////////////////////////////////////////////////////////////////////////////////////////////






////////////////////////////////////////////////////////////////////////////////////////////////////
// 要素保存
////////////////////////////////////////////////////////////////////////////////////////////////////

// canvas要素から
function saveImage(canvas_elem,name="image"){
    let downloadEle = document.createElement("a");
    downloadEle.href = canvas_elem.toDataURL("image/jpg");
    downloadEle.download = name+".jpg";
    downloadEle.click();
}

// html要素から (html2canvas必要)
function saveElemImage(elem,name="image"){
    html2canvas(elem).then(function(canvas) {
        let downloadEle = document.createElement("a");
        downloadEle.href = canvas.toDataURL("image/png");
        downloadEle.download = name+".png";
        downloadEle.click();
        var cdata = canvas.toDataURL();
        var encdata= atob(cdata.replace(/^.*,/, ''));
        var outdata = new Uint8Array(encdata.length);
        for (var i = 0; i < encdata.length; i++) {
            outdata[i] = encdata.charCodeAt(i);
        }
        var blob = new Blob([outdata], ["image/png"]);
        if (window.navigator.msSaveBlob) {
            window.navigator.msSaveOrOpenBlob(blob, fname);
        } else {
            var elem = document.createElement("a");
            elem.href = cdata;
            elem.download = name+".png";
            elem.click();
        }
    });
}

////////////////////////////////////////////////////////////////////////////////////////////////////







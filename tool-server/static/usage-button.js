(function(){
  var ID='prismai-spend-btn';
  function token(){try{return localStorage.getItem('token');}catch(e){return null;}}
  async function isAdmin(t){
    try{var r=await fetch('/api/v1/auths/',{headers:{Authorization:'Bearer '+t}});
        if(!r.ok)return false;var u=await r.json();return !!u&&u.role==='admin';}catch(e){return false;}
  }
  function findNewChat(){
    var els=document.querySelectorAll('a,button');
    for(var i=0;i<els.length;i++){
      var e=els[i],l=((e.getAttribute('aria-label')||'')+' '+(e.textContent||'')).toLowerCase();
      if(l.indexOf('new chat')>=0)return e;
    }
    return null;
  }
  function makeBtn(tpl){
    var b=document.createElement(tpl?tpl.tagName:'button');
    if(tpl&&tpl.className)b.className=tpl.className;
    b.id=ID; b.setAttribute('aria-label','PrismAI spend'); b.title='PrismAI spend';
    b.innerHTML='<span style="display:inline-flex;align-items:center;gap:.5rem;width:100%;white-space:nowrap">\u{1F4B0} Spend</span>';
    if(!tpl)b.style.cssText='display:flex;align-items:center;margin:.2rem .5rem;padding:.5rem .75rem;border:none;border-radius:.5rem;background:transparent;color:inherit;font:inherit;cursor:pointer;opacity:.85';
    b.addEventListener('click',function(ev){ev.preventDefault();ev.stopPropagation();window.open('/usage','_blank');});
    return b;
  }
  function mount(){
    if(document.getElementById(ID))return true;
    var nc=findNewChat();
    if(nc&&nc.parentElement){nc.parentElement.insertBefore(makeBtn(nc),nc.nextSibling);return true;}
    return false;
  }
  async function init(){
    var t=token(); if(!t)return;
    if(!(await isAdmin(t)))return;
    if(mount())return;
    var obs=new MutationObserver(function(){if(mount())obs.disconnect();});
    obs.observe(document.documentElement,{childList:true,subtree:true});
    setTimeout(function(){obs.disconnect();},20000);
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init);else init();
})();

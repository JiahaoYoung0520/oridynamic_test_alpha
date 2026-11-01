/* Browser-only structure viewer. Coordinates and bonds are supplied explicitly. */
(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const catalog = window.ORIDYNAMIC_CASES;
  const groups = ['gt', 'pred'];
  const viewers = [];
  const models = [[], []];
  const cache = new Map();
  const state = {data:null, target:null, ready:false, spinning:false, request:0};
  let syncing = false;
  let homeView = null;
  const colors = [
    {main:'#D280A8', faint:'#E8C2D6', partner:'#E2B1CB', partnerFaint:'#EDD5E3'},
    {main:'#2864AD', faint:'#A6C3E5', partner:'#8063AD', partnerFaint:'#CCBFE0'}
  ];

  function message(text, error=false) {
    byId('status').textContent = text;
    byId('status').hidden = !text;
    byId('status').classList.toggle('error', error);
  }
  function busy(value) {
    state.ready = !value;
    for (const id of ['conformation','view-mode','rotate','reset']) byId(id).disabled = value;
    document.querySelectorAll('#cases button').forEach(b => { b.disabled = value; });
  }
  function loadData(id) {
    if (!cache.has(id)) cache.set(id, new Promise((resolve,reject) => {
      const script = document.createElement('script');
      script.src = `data/${id}.js`;
      script.onload = () => {
        const data = window.ORIDYNAMIC_DATA?.[id];
        if (data) resolve(data); else reject(new Error('Structure data is missing.'));
      };
      script.onerror = () => { cache.delete(id); reject(new Error('Could not load this structure. Reload the page to try again.')); };
      document.head.appendChild(script);
    }));
    return cache.get(id);
  }
  function createViewers() {
    if (viewers.length) return;
    if (!window.$3Dmol) throw new Error('The 3D viewer could not load. You can still download the PyMOL files.');
    for (const [i,id] of ['gt-view','pred-view'].entries()) {
      const viewer = $3Dmol.createViewer(byId(id), {backgroundColor:'white', antialias:true});
      viewer.setProjection('orthographic');
      viewer.setViewChangeCallback(view => {
        if (syncing || !byId('linked').checked || !viewers[1-i]) return;
        syncing = true;
        try { viewers[1-i].setView(view, true); } finally { syncing = false; }
      });
      viewers.push(viewer);
    }
    const observer = new ResizeObserver(() => viewers.forEach(v => v.resize()));
    observer.observe(byId('gt-view')); observer.observe(byId('pred-view'));
  }
  function addModels(data) {
    const neighbors = data.atoms.map(() => []);
    const orders = data.atoms.map(() => []);
    for (const [a,b,order] of data.bonds) {
      neighbors[a].push(b); orders[a].push(order);
      neighbors[b].push(a); orders[b].push(order);
    }
    groups.forEach((group,k) => {
      viewers[k].removeAllModels();
      models[k] = data.groups[group].map(frame => {
        const atoms = data.atoms.map(([atom,resn,resi,chain,elem,protein],i) => ({
          atom,resn,resi,chain,elem,hetflag:!protein,index:i,serial:i+1,
          x:frame.coords[i][0],y:frame.coords[i][1],z:frame.coords[i][2],
          ss:frame.ss[i],bonds:neighbors[i],bondOrder:orders[i]
        }));
        const model = viewers[k].addModel();
        model.addAtoms(atoms);
        return model;
      });
    });
  }
  function styleModels() {
    const data = state.data;
    const all = byId('conformation').value === 'all';
    const chosen = all ? 0 : Number(byId('conformation').value);
    const pocket = data.ligand && byId('view-mode').value === 'pocket';
    groups.forEach((group,k) => {
      models[k].forEach((model,i) => {
        model.setStyle({},{});
        if (!all && i !== chosen) { model.hide(); return; }
        model.show();
        const faint = all && i !== 0;
        const color = faint ? colors[k].faint : colors[k].main;
        if (!pocket) {
          const cartoon = faint ? {style:'trace',thickness:.10,opacity:.4,color} : {color,arrows:true};
          model.setStyle({hetflag:false},{cartoon});
          if (data.id === 'A00C2') {
            model.setStyle({hetflag:false,chain:data.chains[1]},
              {cartoon:{...cartoon,color:faint ? colors[k].partnerFaint : colors[k].partner}});
          }
        } else if (!faint) {
          model.setStyle({index:data.groups[group][i].pocket}, {stick:{radius:.10,color:'#B8C0CD'}});
        }
        if (data.ligand) model.setStyle({hetflag:true}, {stick:{radius:faint ? .10 : .20,color}});
      });
      viewers[k].render();
    });
    byId('display-info').textContent = all ? '8 sampled conformations per ensemble' :
      (chosen === 0 ? 'Representative conformations' : `Conformation ${chosen+1} from each ensemble`);
    byId('caption').textContent = (pocket
      ? 'Ligands are shown in color; nearby protein residues are gray. '
      : all ? 'Solid cartoons show representative conformations; lighter traces show additional samples. '
      : 'Each view shows one sampled conformation. ')
      + 'Both views share the same alignment and scale.'
      + (data.id === 'A00C2' ? ' The partner chain is purple in OriDynamic and pale pink in GT.' : '');
  }
  function fitSharedView() {
    const data = state.data;
    const pocket = data.ligand && byId('view-mode').value === 'pocket';
    const min = [Infinity,Infinity,Infinity], max = [-Infinity,-Infinity,-Infinity];
    for (const group of groups) for (const frame of data.groups[group]) {
      frame.coords.forEach((xyz,i) => {
        if (pocket && data.atoms[i][5]) return;
        for (let a=0;a<3;a++) { min[a]=Math.min(min[a],xyz[a]); max[a]=Math.max(max[a],xyz[a]); }
      });
    }
    const center = min.map((v,i) => (v+max[i])/2);
    let radius = 0;
    for (const group of groups) for (const frame of data.groups[group]) {
      frame.coords.forEach((xyz,i) => {
        if (!pocket || !data.atoms[i][5]) radius = Math.max(radius, Math.hypot(...xyz.map((v,j)=>v-center[j])));
      });
    }
    radius += pocket ? 5 : 2;
    const bounds = [];
    for (let a=0;a<3;a++) for (const sign of [-1,1]) {
      const p = center.slice(); p[a] += sign*radius;
      bounds.push({x:p[0],y:p[1],z:p[2],elem:'C'});
    }
    const v = viewers[0];
    syncing = true;
    try {
      const model = v.addModel(); model.addAtoms(bounds); model.setStyle({},{});
      v.setView([0,0,0,0,0,0,0,1],true);
      v.zoomTo({model:model.getID()});
      v.zoom(1.18);
      v.removeModel(model);
      homeView = v.getView();
      viewers.forEach(viewer => {
        viewer.setSlab(-radius*3-100,radius*3+100);
        viewer.setView(homeView,true); viewer.render();
      });
    } finally { syncing = false; }
  }
  function spin(on) {
    state.spinning = on;
    viewers.forEach(v => v.spin(false));
    if (on) {
      viewers[0].spin('y',.5);
      if (!byId('linked').checked) viewers[1].spin('y',.5);
    }
    byId('rotate').textContent = on ? 'Ⅱ Pause' : '↻ Rotate';
    byId('rotate').setAttribute('aria-pressed',String(on));
  }
  async function selectCase(id) {
    const entry = catalog.find(c => c.id === id) || catalog[0];
    const request = ++state.request;
    busy(true); spin(false); message(`Loading ${entry.id}…`);
    byId('target-name').textContent = entry.id;
    byId('target-info').textContent = `${entry.modality} · ${entry.source} · ${entry.size}`;
    byId('download').href = `downloads/${entry.id}.zip`;
    byId('download').download = `${entry.id}.zip`;
    document.querySelectorAll('#cases button').forEach(b => b.setAttribute('aria-pressed',String(b.dataset.id===entry.id)));
    try {
      const data = await loadData(entry.id);
      if (request !== state.request) return;
      createViewers();
      state.data=data; state.target=entry.id;
      byId('conformation').value='all'; byId('view-mode').value='full';
      byId('pocket-control').hidden=!data.ligand;
      addModels(data); styleModels(); fitSharedView();
      busy(false); message('');
    } catch (err) {
      if (request !== state.request) return;
      state.ready=false;
      document.querySelectorAll('#cases button').forEach(b => { b.disabled=false; });
      message(err.message || 'Could not initialize 3D graphics. Try a browser with WebGL support.',true);
    }
  }
  for (const c of catalog) {
    const button=document.createElement('button'); button.type='button'; button.dataset.id=c.id;
    const name=document.createElement('strong'); name.textContent=c.id;
    const subtitle=document.createElement('small'); subtitle.textContent=c.source+' · '+c.size.replace(' + 22 ligand heavy atoms',' + ligand');
    button.append(name,subtitle); button.setAttribute('aria-pressed','false');
    button.addEventListener('click',() => {
      const hash=`case=${c.id}`;
      if (location.hash.slice(1)===hash) selectCase(c.id); else location.hash=hash;
    });
    byId('cases').appendChild(button);
  }
  byId('conformation').addEventListener('change',() => { if (state.ready) styleModels(); });
  byId('view-mode').addEventListener('change',() => {
    if (!state.ready) return;
    spin(false); styleModels(); fitSharedView();
  });
  byId('rotate').addEventListener('click',() => { if (state.ready) spin(!state.spinning); });
  byId('reset').addEventListener('click',() => {
    if (!state.ready) return;
    spin(false); viewers.forEach(v => v.setView(homeView,true));
  });
  byId('linked').addEventListener('change',() => {
    if (byId('linked').checked && viewers.length===2) viewers[1].setView(viewers[0].getView(),true);
    spin(state.spinning);
  });
  document.addEventListener('visibilitychange',() => { if (document.hidden) spin(false); });
  const fromHash=() => selectCase(new URLSearchParams(location.hash.slice(1)).get('case'));
  window.addEventListener('hashchange',fromHash);
  // Expose the visible state for reproducible browser checks and inspection.
  window.oridynamicViewer={state,viewers,models};
  fromHash();
})();

(this["webpackJsonpperceptron-visualization"]=this["webpackJsonpperceptron-visualization"]||[]).push([[0],{13:function(e,t,a){},14:function(e,t,a){},15:function(e,t,a){"use strict";a.r(t);var l=a(0),r=a.n(l),n=a(3),c=a.n(n);a(13),a(14);var o=()=>{const e=[{inputs:[1,1],output:1},{inputs:[1,0],output:0},{inputs:[0,1],output:0},{inputs:[0,0],output:0}],[t,a]=Object(l.useState)([1,1]),[n,c]=Object(l.useState)(-3),[o,s]=Object(l.useState)(!1),[i,m]=Object(l.useState)(0),[d,u]=Object(l.useState)(1),[E,x]=Object(l.useState)(0),[f,p]=Object(l.useState)(null),[b,h]=Object(l.useState)(0),[g,N]=Object(l.useState)([]),[v,y]=Object(l.useState)(1),[w,k]=Object(l.useState)(!1),[F,S]=Object(l.useState)(!1),[j,A]=Object(l.useState)("feedforward"),[W,$]=Object(l.useState)(new Array(e.length).fill(null)),z=(e,t,a)=>e[0]*t[0]+e[1]*t[1]+a>=0?1:0,C=()=>{if(w)return void s(!1);const l=e[E];if("feedforward"===j){const e=(r=l.inputs,o=t,u=n,r[0]*o[0]+r[1]*o[1]+u);h(e);const a=e>=0?1:0;p(a),$(e=>{const t=[...e];return t[E]=a,t}),A("backprop")}else{const r=f,o=l.output-r;if(F)N(e=>[...e,`\xc9poca extra ${i}, Muestra [${l.inputs}] \u2192 ${l.output}, Predicci\xf3n: ${r}, Error: ${o} (Pesos ya finales)`].slice(-10));else if(0!==o){const e=[t[0]+d*o*l.inputs[0],t[1]+d*o*l.inputs[1]],s=n+d*o;a(e),c(s),N(t=>[...t,`\xc9poca ${i}, Muestra [${l.inputs}] \u2192 ${l.output}, Predicci\xf3n: ${r}, Error: ${o}, Pesos: [${e.map(e=>e.toFixed(2))}], Bias: ${s.toFixed(2)}`].slice(-10))}else N(e=>[...e,`\xc9poca ${i}, Muestra [${l.inputs}] \u2192 ${l.output}, Predicci\xf3n: ${r}, Error: ${o}, Pesos: [${t.map(e=>e.toFixed(2))}], Bias: ${n.toFixed(2)}`].slice(-10));let s=(E+1)%e.length;if(x(s),0===s){m(i+1);const a=((t,a)=>{let l=0;for(let r=0;r<e.length;r++){const n=e[r],c=z(n.inputs,t,a);l+=Math.abs(n.output-c)}return l})(t,n);y(a),F||0!==a||(S(!0),N(e=>[...e,"\xa1Convergencia lograda! Ejecutando una \xe9poca adicional de demostraci\xf3n..."])),F&&(k(!0),N(e=>[...e,"Entrenamiento completado. El perceptr\xf3n ha aprendido la funci\xf3n AND."]))}A("feedforward"),p(null),h(0)}var r,o,u};Object(l.useEffect)(()=>{if(w){const a=e.map(e=>z(e.inputs,t,n));$(a)}},[w,t,n,e]),Object(l.useEffect)(()=>{let e;return o&&(e=setInterval(C,1e3)),()=>clearInterval(e)},[o,E,t,n,i,d,j]);return r.a.createElement("div",{className:"flex flex-col items-center max-w-4xl mx-auto p-4 bg-gray-50 rounded-lg shadow"},r.a.createElement("h2",{className:"text-2xl font-bold mb-6"},"Animaci\xf3n de Perceptr\xf3n - Compuerta AND"),r.a.createElement("div",{className:"w-full bg-white rounded-lg shadow p-4 mb-4"},r.a.createElement("h3",{className:"font-bold mb-4 text-center"},"Representaci\xf3n del Modelo"),r.a.createElement("div",{className:"flex justify-center mb-4"},(()=>{const a=e[E],l=a.inputs[0],c=a.inputs[1],o=t[0],s=t[1],i=n;return r.a.createElement("svg",{width:"500",height:"280",viewBox:"0 0 500 280"},r.a.createElement("defs",null,r.a.createElement("marker",{id:"arrowhead",markerWidth:"10",markerHeight:"7",refX:"0",refY:"3.5",orient:"auto"},r.a.createElement("polygon",{points:"0 0, 10 3.5, 0 7",fill:"#000"}))),r.a.createElement("g",{className:"inputs"},r.a.createElement("circle",{cx:"50",cy:"70",r:"20",fill:"#E3F2FD",stroke:"#2196F3",strokeWidth:"2"}),r.a.createElement("text",{x:"50",y:"75",fontSize:"14",textAnchor:"middle"},"x\u2081=",l),r.a.createElement("circle",{cx:"50",cy:"160",r:"20",fill:"#E3F2FD",stroke:"#2196F3",strokeWidth:"2"}),r.a.createElement("text",{x:"50",y:"165",fontSize:"14",textAnchor:"middle"},"x\u2082=",c),r.a.createElement("circle",{cx:"50",cy:"250",r:"20",fill:"#FCE4EC",stroke:"#E91E63",strokeWidth:"2"}),r.a.createElement("text",{x:"50",y:"255",fontSize:"14",textAnchor:"middle"},"1")),r.a.createElement("g",{className:"summation"},r.a.createElement("circle",{cx:"240",cy:"160",r:"30",fill:"#FFF3E0",stroke:"#FF9800",strokeWidth:"2"}),r.a.createElement("text",{x:"240",y:"170",fontSize:"24",textAnchor:"middle"},"\u03a3"),r.a.createElement("text",{x:"240",y:"200",fontSize:"12",textAnchor:"middle"},"Suma: ",b.toFixed(2))),r.a.createElement("g",{className:"activation"},r.a.createElement("circle",{cx:"360",cy:"160",r:"30",fill:"#F1F8E9",stroke:"#8BC34A",strokeWidth:"2"}),r.a.createElement("svg",{x:"330",y:"130",width:"60",height:"60",viewBox:"0 0 60 60",overflow:"visible"},r.a.createElement("line",{x1:"0",y1:"30",x2:"20",y2:"30",stroke:"black",strokeWidth:"2"}),r.a.createElement("line",{x1:"20",y1:"30",x2:"20",y2:"10",stroke:"black",strokeWidth:"2"}),r.a.createElement("line",{x1:"20",y1:"10",x2:"60",y2:"10",stroke:"black",strokeWidth:"2"}),r.a.createElement("text",{x:"30",y:"5",fontSize:"10",textAnchor:"middle"},"f(x)"),r.a.createElement("line",{x1:"0",y1:"50",x2:"60",y2:"50",stroke:"#777",strokeWidth:"0.5"})," ",r.a.createElement("line",{x1:"20",y1:"0",x2:"20",y2:"50",stroke:"#777",strokeWidth:"0.5"})," ",r.a.createElement("circle",{cx:b<0?10:40,cy:b<0?30:10,r:"3",fill:"red"}))),r.a.createElement("g",{className:"output"},r.a.createElement("circle",{cx:"450",cy:"160",r:"20",fill:"#FFEBEE",stroke:"#F44336",strokeWidth:1===f?"4":"2"}),r.a.createElement("text",{x:"450",y:"165",fontSize:"16",textAnchor:"middle"},null===f?"?":f)),r.a.createElement("g",{className:"connections"},r.a.createElement("line",{x1:"70",y1:"70",x2:"212",y2:"150",stroke:o>=0?"#4CAF50":"#F44336",strokeWidth:1.5*Math.abs(o)+1}),r.a.createElement("text",{x:"130",y:"100",fontSize:"12",fontWeight:"bold"},o.toFixed(2)),r.a.createElement("line",{x1:"70",y1:"160",x2:"210",y2:"160",stroke:s>=0?"#4CAF50":"#F44336",strokeWidth:1.5*Math.abs(s)+1}),r.a.createElement("text",{x:"130",y:"150",fontSize:"12",fontWeight:"bold"},s.toFixed(2)),r.a.createElement("line",{x1:"70",y1:"250",x2:"212",y2:"170",stroke:i>=0?"#4CAF50":"#F44336",strokeWidth:1.5*Math.abs(i)+1}),r.a.createElement("text",{x:"130",y:"220",fontSize:"12",fontWeight:"bold"},i.toFixed(2)),r.a.createElement("line",{x1:"270",y1:"160",x2:"330",y2:"160",stroke:"#000",strokeWidth:"2",markerEnd:"url(#arrowhead)"}),r.a.createElement("line",{x1:"390",y1:"160",x2:"430",y2:"160",stroke:"#000",strokeWidth:"2",markerEnd:"url(#arrowhead)"})),("backprop"===j||w)&&r.a.createElement("g",{className:"target"},r.a.createElement("rect",{x:"370",y:"80",width:"120",height:"40",fill:"#F5F5F5",stroke:"#9E9E9E",strokeWidth:"1",rx:"5"}),r.a.createElement("text",{x:"430",y:"100",fontSize:"12",textAnchor:"middle"},"Target: ",a.output),r.a.createElement("text",{x:"430",y:"115",fontSize:"12",textAnchor:"middle"},"Error: ",a.output-f)),r.a.createElement("rect",{x:"240",y:"30",width:"160",height:"30",fill:"feedforward"===j?"#E3F2FD":"#FFEBEE",stroke:"#9E9E9E",strokeWidth:"1",rx:"5"}),r.a.createElement("text",{x:"320",y:"50",fontSize:"14",fontWeight:"bold",textAnchor:"middle"},"Fase: ","feedforward"===j?"Forward":"Backprop"))})())),r.a.createElement("div",{className:"w-full bg-white rounded-lg shadow p-4 mb-4"},r.a.createElement("h3",{className:"font-bold mb-2"},"Estado del Entrenamiento"),r.a.createElement("div",{className:"flex flex-col space-y-2"},r.a.createElement("div",{className:"flex items-center"},r.a.createElement("div",{className:"w-32"},"Estado:"),r.a.createElement("div",{className:"font-medium "+(F?w?"text-green-600":"text-blue-600":"text-amber-600")},w?"\u2713 Entrenamiento completado":F?"\u2713 Convergencia lograda - Ejecutando \xe9poca adicional":"\u27f3 Entrenando...")),r.a.createElement("div",{className:"flex items-center"},r.a.createElement("div",{className:"w-32"},"Error total:"),r.a.createElement("div",{className:"font-mono"},v)),r.a.createElement("div",{className:"flex items-center"},r.a.createElement("div",{className:"w-32"},"\xc9poca actual:"),r.a.createElement("div",{className:"font-mono"},i)))),r.a.createElement("div",{className:"flex flex-wrap justify-center gap-4 w-full"},r.a.createElement("div",{className:"bg-white rounded-lg shadow p-4 flex-1"},r.a.createElement("h3",{className:"font-bold mb-2"},"Estado Actual"),r.a.createElement("div",{className:"grid grid-cols-2 gap-2 mb-4"},r.a.createElement("div",null,"Fase actual:"),r.a.createElement("div",{className:"font-mono "+("feedforward"===j?"text-blue-500":"text-red-500")},"feedforward"===j?"Forward-propagation":"Back-propagation"),r.a.createElement("div",null,"\xc9poca:"),r.a.createElement("div",{className:"font-mono"},i),r.a.createElement("div",null,"Pesos:"),r.a.createElement("div",{className:"font-mono"},"[",t.map(e=>e.toFixed(3)).join(", "),"]"),r.a.createElement("div",null,"Bias:"),r.a.createElement("div",{className:"font-mono"},n.toFixed(3)),r.a.createElement("div",null,"Tasa de aprendizaje:"),r.a.createElement("div",{className:"font-mono"},d),r.a.createElement("div",null,"Error total:"),r.a.createElement("div",{className:"font-mono"},v),r.a.createElement("div",null,"Muestra actual:"),r.a.createElement("div",{className:"font-mono"},"[",e[E].inputs.join(", "),"] \u2192 ",e[E].output),r.a.createElement("div",null,"Salida del sumador:"),r.a.createElement("div",{className:"font-mono"},b.toFixed(3)),r.a.createElement("div",null,"Predicci\xf3n:"),r.a.createElement("div",{className:"font-mono "+(null===f?"text-gray-500":f===e[E].output?"text-green-500":"text-red-500")},null===f?"...":f))),r.a.createElement("div",{className:"bg-white rounded-lg shadow p-4 flex-1"},r.a.createElement("h3",{className:"font-bold mb-2"},"Tabla de Verdad AND"),r.a.createElement("table",{className:"w-full table-auto border-collapse mb-4"},r.a.createElement("thead",null,r.a.createElement("tr",{className:"bg-gray-100"},r.a.createElement("th",{className:"border p-1"},"X\u2081"),r.a.createElement("th",{className:"border p-1"},"X\u2082"),r.a.createElement("th",{className:"border p-1"},"Esperado"),r.a.createElement("th",{className:"border p-1"},"Predicci\xf3n"))),r.a.createElement("tbody",null,e.map((e,t)=>r.a.createElement("tr",{key:t,className:t===E?"bg-yellow-100":""},r.a.createElement("td",{className:"border p-1 text-center"},e.inputs[0]),r.a.createElement("td",{className:"border p-1 text-center"},e.inputs[1]),r.a.createElement("td",{className:"border p-1 text-center"},e.output),r.a.createElement("td",{className:"border p-1 text-center "+(null===W[t]?"":W[t]===e.output?"text-green-500 font-bold":"text-red-500 font-bold")},null!==W[t]?W[t]:""))))))),r.a.createElement("div",{className:"w-full bg-white rounded-lg shadow p-4 mb-4"},r.a.createElement("h3",{className:"font-bold mb-2"},"Registro de Actualizaci\xf3n de Pesos"),r.a.createElement("div",{className:"h-32 overflow-y-auto bg-gray-100 p-2 rounded text-xs font-mono"},0===g.length?r.a.createElement("p",null,"Inicie el entrenamiento para ver el registro..."):g.map((e,t)=>r.a.createElement("p",{key:t},e)))),r.a.createElement("div",{className:"flex gap-2 mb-4"},r.a.createElement("button",{onClick:()=>s(!o),className:`px-4 py-2 rounded ${w?"bg-gray-300":o?"bg-red-500 hover:bg-red-600":"bg-green-500 hover:bg-green-600"} text-white`,disabled:w},o?"Pausar":"Auto"),r.a.createElement("button",{onClick:C,className:`px-4 py-2 rounded ${"feedforward"===j?"bg-blue-500 hover:bg-blue-600":"bg-orange-500 hover:bg-orange-600"} text-white`,disabled:o||w},"feedforward"===j?"Forward \u25b6":"Backprop \u25b6"),r.a.createElement("button",{onClick:()=>{a([1,1]),c(-3),m(0),x(0),p(null),h(0),N([]),y(1),k(!1),S(!1),s(!1),A("feedforward"),$(new Array(e.length).fill(null))},className:"px-4 py-2 rounded bg-blue-500 hover:bg-blue-600 text-white"},"Reiniciar")),r.a.createElement("div",{className:"w-full bg-white rounded-lg shadow p-4"},r.a.createElement("h3",{className:"font-bold mb-2"},"Controles"),r.a.createElement("div",{className:"flex items-center gap-4 mb-2"},r.a.createElement("label",null,"Tasa de aprendizaje:"),r.a.createElement("input",{type:"range",min:"0.01",max:"1",step:"0.01",value:d,onChange:e=>u(parseFloat(e.target.value)),className:"w-32"}),r.a.createElement("span",{className:"font-mono"},d.toFixed(2))),r.a.createElement("div",{className:"mt-4"},r.a.createElement("h4",{className:"font-bold"},"Leyenda:"),r.a.createElement("ul",{className:"text-sm"},r.a.createElement("li",{className:"flex items-center"},r.a.createElement("span",{className:"inline-block w-3 h-3 bg-green-500 rounded-full mr-1"})," Peso positivo"),r.a.createElement("li",{className:"flex items-center"},r.a.createElement("span",{className:"inline-block w-3 h-3 bg-red-500 rounded-full mr-1"})," Peso negativo"),r.a.createElement("li",{className:"flex items-center"},r.a.createElement("span",{className:"inline-block w-3 h-3 bg-yellow-500 rounded-full mr-1"})," Sumador"),r.a.createElement("li",{className:"flex items-center"},r.a.createElement("span",{className:"inline-block w-3 h-3 bg-green-200 rounded-full mr-1"})," Funci\xf3n de activaci\xf3n (escal\xf3n)")))))};var s=function(){return r.a.createElement("div",{className:"App"},r.a.createElement(o,null))};var i=e=>{e&&e instanceof Function&&a.e(3).then(a.bind(null,16)).then(t=>{let{getCLS:a,getFID:l,getFCP:r,getLCP:n,getTTFB:c}=t;a(e),l(e),r(e),n(e),c(e)})};c.a.createRoot(document.getElementById("root")).render(r.a.createElement(r.a.StrictMode,null,r.a.createElement(s,null))),i()},4:function(e,t,a){e.exports=a(15)}},[[4,1,2]]]);
//# sourceMappingURL=main.d878cb76.chunk.js.map
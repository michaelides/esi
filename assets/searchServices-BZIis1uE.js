import{i as c}from"./index-BfbaazB6.js";const o={wikipedia:{baseUrl:"https://en.wikipedia.org/w/api.php",params:{action:"query",format:"json",prop:"extracts|pageimages",exintro:!0,explaintext:!0,exsentences:3,piprop:"thumbnail",pithumbsize:200}},semanticScholar:{baseUrl:"https://api.semanticscholar.org/graph/v1/paper/search",fields:"title,authors,year,abstract,url,venue,citationCount"}},h=async(t,i=5)=>{try{return(await c({kind:"tavily",query:t,limit:i})).results?.map(r=>({title:r.title,snippet:r.content,url:r.url,source:"Tavily",score:r.score}))||[]}catch(e){throw console.error("Tavily search error:",e),new Error(`Web search failed: ${e.message}`)}},u=async t=>{try{const i=new URLSearchParams({action:"query",format:"json",formatversion:"2",origin:"*",prop:"extracts|pageimages|info",exintro:!0,explaintext:!0,exsentences:3,piprop:"thumbnail",pithumbsize:200,inprop:"url",generator:"search",gsrsearch:t,gsrlimit:5}),e=await fetch(`${o.wikipedia.baseUrl}?${i}`);if(!e.ok)throw new Error(`Wikipedia API error: ${e.status}`);return((await e.json()).query?.pages||[]).map(a=>({title:a.title,snippet:a.extract||"No description available",url:a.fullurl||`https://en.wikipedia.org/wiki/${encodeURIComponent(a.title.replace(/ /g,"_"))}`,source:"Wikipedia",thumbnail:a.thumbnail?.source}))}catch(i){throw console.error("Wikipedia search error:",i),new Error(`Wikipedia search failed: ${i.message}`)}},m=async(t,i=5)=>{try{const e=new URLSearchParams({query:t,limit:Math.min(Math.max(1,i),20),fields:o.semanticScholar.fields}),r=await fetch(`${o.semanticScholar.baseUrl}?${e}`);if(!r.ok)throw console.error("Semantic Scholar API error:",r.status,r.statusText),new Error(`Semantic Scholar API error: ${r.status}`);return((await r.json()).data??[]).map(a=>({title:a.title||"No title",snippet:a.abstract||"No abstract available",url:a.url,source:"Semantic Scholar",authors:(a.authors??[]).map(s=>s.name).filter(Boolean).join(", ")||"Unknown authors",year:String(a.year??"Unknown year"),venue:a.venue||"Unknown venue",citationCount:a.citationCount||0}))}catch(e){throw console.error("Semantic Scholar search error:",e),new Error(`Academic search failed: ${e.message}`)}},p=(t,i)=>{if(!t||t.length===0)return"No results found for your search query.";let e=`## Search Results (${i})

`;return t.forEach((r,n)=>{e+=`### ${n+1}. ${r.title}
`,r.snippet&&(e+=`${r.snippet}

`),r.authors&&i==="Semantic Scholar"&&(e+=`**Authors:** ${r.authors}  
`,e+=`**Year:** ${r.year}  
`,e+=`**Venue:** ${r.venue}  
`,e+=`**Citations:** ${r.citationCount}  
`),r.thumbnail&&(e+=`![Thumbnail](${r.thumbnail})

`),e+=`[🔗 View Source](${r.url})

`,e+=`---

`}),e},d=(t,i)=>{if(!t||t.length===0)return{valid:!1,error:"Please provide a search query"};let e="",r=5;if(i==="ss"){const n=t[t.length-1],a=parseInt(n);!isNaN(a)&&a>0&&a<=20?(r=a,e=t.slice(0,-1).join(" ")):e=t.join(" ")}else e=t.join(" ");return e.trim()?{valid:!0,query:e.trim(),limit:r}:{valid:!1,error:"Please provide a valid search query"}};export{p as formatSearchResults,m as searchSemanticScholar,h as searchTavily,u as searchWikipedia,d as validateSearchParams};

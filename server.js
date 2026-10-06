
const express = require("express");
const fs = require("fs");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000;
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

const DATA = path.join(__dirname, "data", "credit_data.csv");

function sigmoid(z){ return 1/(1+Math.exp(-Math.max(-30,Math.min(30,z)))); }

function loadCSV(){
  const lines = fs.readFileSync(DATA,"utf8").trim().split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).map(line=>{
    const v=line.split(",");
    const o={}; headers.forEach((h,i)=>o[h]=Number(v[i]));
    return o;
  });
}

function trainModel(rows){
  const features=["monthly_sales","avg_order_value","inventory_turnover","return_rate","late_payments","business_age","digital_share"];
  const X=rows.map(r=>features.map(f=>r[f]));
  const y=rows.map(r=>r.defaulted);
  const means=features.map((_,j)=>X.reduce((s,r)=>s+r[j],0)/X.length);
  const stds=features.map((_,j)=>{
    const m=means[j];
    return Math.sqrt(X.reduce((s,r)=>s+(r[j]-m)**2,0)/X.length)||1;
  });
  const Z=X.map(r=>r.map((v,j)=>(v-means[j])/stds[j]));
  let w=new Array(features.length).fill(0), b=0, lr=.08;
  for(let epoch=0;epoch<1800;epoch++){
    let gw=new Array(features.length).fill(0), gb=0;
    for(let i=0;i<Z.length;i++){
      const p=sigmoid(Z[i].reduce((s,v,j)=>s+w[j]*v,b));
      const e=p-y[i];
      gb+=e;
      for(let j=0;j<w.length;j++) gw[j]+=e*Z[i][j];
    }
    for(let j=0;j<w.length;j++) w[j]-=lr*gw[j]/Z.length;
    b-=lr*gb/Z.length;
  }
  let correct=0;
  rows.forEach((r,i)=>{
    const z=Z[i].reduce((s,v,j)=>s+w[j]*v,b);
    if((sigmoid(z)>=.5)===(y[i]===1)) correct++;
  });
  return {features,means,stds,w,b,accuracy:correct/rows.length};
}

let rows=loadCSV();
let model=trainModel(rows);

function predict(input){
  const z=model.features.reduce((s,f,j)=>s+wSafe(model.w[j])*((Number(input[f])-model.means[j])/model.stds[j]),model.b);
  const p=sigmoid(z);
  return {default_probability:p, risk:p>=.65?"High":p>=.35?"Medium":"Low", suggested_limit:Math.round(Math.max(25000,Math.min(500000,(1-p)*500000))/5000)*5000};
}
function wSafe(v){return Number.isFinite(v)?v:0;}

app.get("/api/summary",(req,res)=>{
  const avgSales=rows.reduce((s,r)=>s+r.monthly_sales,0)/rows.length;
  const avgReturn=rows.reduce((s,r)=>s+r.return_rate,0)/rows.length;
  const highRisk=rows.filter(r=>predict(r).risk==="High").length;
  res.json({
    customers: 1248, activeLoans: 386, portfolio: 48750000,
    approvalRate: 72.4, avgSales: Math.round(avgSales),
    returnRate: +(avgReturn*100).toFixed(1), highRisk,
    deadstockValue: 6840000, predictedLossReduction: 18.6,
    modelAccuracy:+(model.accuracy*100).toFixed(1)
  });
});

app.get("/api/products",(req,res)=>{
  res.json([
    {id:"AW-101",name:"CloudFit Oversized Tee",category:"T-Shirts",price:899,stock:428,sold30:612,returnRate:4.2,status:"Hot"},
    {id:"AW-203",name:"Everyday Co-ord Set",category:"Co-ords",price:1899,stock:166,sold30:284,returnRate:7.1,status:"Hot"},
    {id:"AW-311",name:"SoftForm Straight Jeans",category:"Denim",price:2199,stock:310,sold30:144,returnRate:13.8,status:"Watch"},
    {id:"AW-412",name:"Studio Shirt",category:"Shirts",price:1599,stock:502,sold30:88,returnRate:18.4,status:"Clearance"},
    {id:"AW-509",name:"DailyFlex Joggers",category:"Bottoms",price:1299,stock:245,sold30:221,returnRate:6.3,status:"Healthy"}
  ]);
});

app.post("/api/train",(req,res)=>{
  rows=loadCSV();
  model=trainModel(rows);
  res.json({message:"Model retrained successfully",rows:rows.length,accuracy:+(model.accuracy*100).toFixed(1),features:model.features});
});

app.post("/api/predict",(req,res)=>{
  try{res.json(predict(req.body));}catch(e){res.status(400).json({error:e.message});}
});

app.post("/api/chat",(req,res)=>{
  const q=String(req.body.message||"").toLowerCase();
  let answer;
  if(q.includes("loan")||q.includes("credit")||q.includes("borrow"))
    answer="AuraWear Credit uses sales, inventory turnover, return rate, repayment behaviour, business age and digital-sales share to estimate risk. Low-risk merchants can unlock a larger working-capital limit.";
  else if(q.includes("return")||q.includes("size"))
    answer="The fit engine flags products with high return rates and recommends size guidance, product-detail improvements and inventory actions. In this demo, products above 12% returns are placed on Watch/Clearance.";
  else if(q.includes("inventory")||q.includes("stock")||q.includes("dead"))
    answer="The inventory engine identifies slow-moving SKUs and recommends markdowns, bundles or credit-linked restocking instead of blind reordering.";
  else if(q.includes("business")||q.includes("model"))
    answer="The model earns from clothing gross margin, merchant financing interest/fees, subscription analytics and partner commissions. Credit is repaid from future merchant cash flow.";
  else if(q.includes("train")||q.includes("ai"))
    answer="Open AI Lab to retrain the risk model on the CSV dataset. The backend trains a logistic-regression classifier and reports accuracy.";
  else
    answer="I can help with credit, inventory, returns, AI training, products or the AuraWear business model. Try: “How does credit work?”";
  res.json({answer});
});

app.get("*",(req,res)=>res.sendFile(path.join(__dirname,"public","index.html")));
app.listen(PORT,()=>console.log(`AuraWear Credit AI running on http://localhost:${PORT}`));

// === main.cpp (inference + accuracy + AP@0.50 + CSV metrics; quiet per-image) ===
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <new>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <dirent.h>
#include <errno.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>
#include <ctime>
#include <thread>
#include <opencv2/dnn.hpp>


// ---------- Config ----------
static constexpr int   INPUT_W    = 640;
static constexpr int   INPUT_H    = 640;
static constexpr float CONF_THR   = 0.25f;   // prediction confidence threshold
static constexpr float IOU_THR    = 0.45f;   // NMS IoU for predictions
static constexpr float MATCH_IOU  = 0.50f;   // IoU threshold to count as TP (evaluation)
static constexpr double PX_TO_UM  = 0.5;     // pixel to micron (for overlay text only)
static constexpr bool  SAVE_VIS   = false;    // write *_vis.jpg
static constexpr bool  VERBOSE_PER_IMAGE = false; // print per-image "[Saved]" lines
static constexpr bool  COMPUTE_MAP = true;   // compute AP@mIoU
static constexpr float MAP_IOU     = MATCH_IOU; // AP@0.50

// ---------- Basic structs ----------
struct Detection { cv::Rect box; float score; int class_id; };
struct GTBox     { cv::Rect box; int class_id; };

// ---------- Small utils ----------
static inline float clampf(float v, float lo, float hi){ return v<lo?lo:(v>hi?hi:v); }
static float IoU_rect(const cv::Rect& a, const cv::Rect& b){
    float inter = (float)((a & b).area());
    float uni   = (float)(a.area() + b.area() - (a & b).area());
    return uni > 0.f ? inter / uni : 0.f;
}
static std::string to_lower(std::string s){ for(char& c:s) c=(char)std::tolower((unsigned char)c); return s; }
static void ensure_dir(const std::string& dir){
    if(dir.empty()) return;
    struct stat st;
    if(stat(dir.c_str(),&st)==0) return;
    if(mkdir(dir.c_str(),0755)!=0 && errno!=EEXIST)
        std::cerr<<"Warning: mkdir failed for "<<dir<<" ("<<strerror(errno)<<")\n";
}
static std::string join_path(const std::string& a,const std::string& b){
    if(a.empty()) return b;
    return a.back()=='/'? a+b : a+"/"+b;
}
static std::string basename_no_ext(const std::string& p){
    size_t s=p.find_last_of('/');
    std::string base=(s==std::string::npos)?p:p.substr(s+1);
    size_t d=base.find_last_of('.');
    return d==std::string::npos? base : base.substr(0,d);
}
static std::vector<std::string> list_images(const std::string& dir, size_t limit=100000){
    std::vector<std::string> out; DIR* dp=opendir(dir.c_str());
    if(!dp){ std::cerr<<"Could not open images dir: "<<dir<<" ("<<strerror(errno)<<")\n"; return out; }
    dirent* ent;
    while((ent=readdir(dp))!=nullptr){
        if(ent->d_name[0]=='.') continue;
        std::string name=ent->d_name; size_t dot=name.find_last_of('.');
        std::string ext=to_lower(dot==std::string::npos? "" : name.substr(dot));
        if(ext==".jpg"||ext==".jpeg"||ext==".png"||ext==".bmp"||ext==".tif"||ext==".tiff"){
            std::string full=dir; if(!full.empty()&&full.back()!='/') full+='/'; full+=name; out.push_back(full);
            if(out.size()>=limit) break;
        }
    }
    closedir(dp);
    return out;
}

// ---------- NMS ----------
static float IoU(const cv::Rect& a, const cv::Rect& b){ return IoU_rect(a,b); }
static std::vector<int> NMS(const std::vector<Detection>& dets, float iou_thr){
    std::vector<int> idxs(dets.size());
    for(size_t i=0;i<idxs.size();++i) idxs[i]=(int)i;
    std::sort(idxs.begin(), idxs.end(), [&](int i,int j){return dets[i].score>dets[j].score;});
    std::vector<int> keep; std::vector<char> removed(dets.size(),0);
    for(size_t m=0;m<idxs.size();++m){
        int i=idxs[m]; if(removed[i]) continue; keep.push_back(i);
        for(size_t n=m+1;n<idxs.size();++n){
            int j=idxs[n]; if(removed[j]) continue;
            if(IoU(dets[i].box,dets[j].box)>iou_thr) removed[j]=1;
        }
    }
    return keep;
}

// ---------- Letterbox helpers (Ultralytics style) ----------
struct LetterboxInfo{ float scale; int pad_x; int pad_y; };
static LetterboxInfo letterbox_bgr_to_rgb_640(const cv::Mat& src, cv::Mat& dst640_rgb){
    const int in_w=src.cols, in_h=src.rows;
    const float r=std::min((float)INPUT_W/in_w, (float)INPUT_H/in_h);
    const int new_w=std::max(1,(int)std::round(in_w*r));
    const int new_h=std::max(1,(int)std::round(in_h*r));
    const int pad_x=(INPUT_W-new_w)/2, pad_y=(INPUT_H-new_h)/2;
    cv::Mat resized; cv::resize(src,resized,cv::Size(new_w,new_h),0,0,cv::INTER_LINEAR);
    cv::Mat canvas(INPUT_H,INPUT_W,CV_8UC3,cv::Scalar(114,114,114));
    resized.copyTo(canvas(cv::Rect(pad_x,pad_y,new_w,new_h)));
    cv::cvtColor(canvas,dst640_rgb,cv::COLOR_BGR2RGB);
    return {r,pad_x,pad_y};
}
static inline void undo_letterbox_xyxy(float& x1,float& y1,float& x2,float& y2,const LetterboxInfo& L,int ow,int oh){
    x1=(x1-L.pad_x)/L.scale; y1=(y1-L.pad_y)/L.scale; x2=(x2-L.pad_x)/L.scale; y2=(y2-L.pad_y)/L.scale;
    x1=clampf(x1,0.f,(float)(ow-1)); y1=clampf(y1,0.f,(float)(oh-1)); x2=clampf(x2,0.f,(float)(ow-1)); y2=clampf(y2,0.f,(float)(oh-1));
}

// ---------- Read YOLO TXT labels (normalized) ----------
static std::vector<GTBox> load_gt_yolo(const std::string& label_path, int img_w, int img_h){
    std::vector<GTBox> gts;
    std::ifstream in(label_path);
    if(!in.is_open()) return gts; // treat as no objects
    std::string line;
    while(std::getline(in, line)){
        if(line.empty()) continue;
        std::istringstream iss(line);
        int cls; float xc, yc, w, h;
        if(!(iss >> cls >> xc >> yc >> w >> h)) continue;
        float x1 = (xc - w/2.f) * img_w;
        float y1 = (yc - h/2.f) * img_h;
        float x2 = (xc + w/2.f) * img_w;
        float y2 = (yc + h/2.f) * img_h;
        x1 = clampf(x1, 0.f, (float)(img_w-1));
        y1 = clampf(y1, 0.f, (float)(img_h-1));
        x2 = clampf(x2, 0.f, (float)(img_w-1));
        y2 = clampf(y2, 0.f, (float)(img_h-1));
        cv::Rect r(cv::Point2f(x1,y1), cv::Point2f(x2,y2));
        if(r.width>0 && r.height>0) gts.push_back({r, cls});
    }
    return gts;
}

// ---------- AP helpers ----------
struct PRAccum {
    std::vector<float> scores;   // prediction scores (desc per image)
    std::vector<int>   tp_flags; // 1 if TP, 0 if FP (aligned with scores)
    long tp=0, fp=0, fn=0;
    double sumIoU_TP=0.0;
};

// Per-image matching that also collects TP/FP flags for PR curve
static PRAccum evaluate_for_map(const std::vector<Detection>& preds,
                                const std::vector<GTBox>& gts,
                                bool match_by_class)
{
    PRAccum out;
    // sort preds by score desc (valid PR)
    std::vector<int> order(preds.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a,int b){ return preds[a].score > preds[b].score; });

    std::vector<char> matched(gts.size(), 0);

    for (int pi : order) {
        const auto& p = preds[pi];
        float best_iou = 0.f; int best_idx = -1;
        for (int i=0; i<(int)gts.size(); ++i) {
            if (matched[i]) continue;
            if (match_by_class && (p.class_id != gts[i].class_id)) continue;
            float iou = IoU_rect(p.box, gts[i].box);
            if (iou > best_iou) { best_iou = iou; best_idx = i; }
        }

        out.scores.push_back(p.score);
        if (best_idx != -1 && best_iou >= MAP_IOU) {
            matched[best_idx] = 1;
            out.tp_flags.push_back(1);
            out.tp += 1;
            out.sumIoU_TP += best_iou;
        } else {
            out.tp_flags.push_back(0);
            out.fp += 1;
        }
    }

    for (char m : matched) if (!m) out.fn += 1; // unmatched GTs are FN
    return out;
}

// Compute AP using 101-point interpolation on the precision–recall curve
static double compute_ap_101(const std::vector<float>& scores_desc,
                             const std::vector<int>& tp_flags_desc,
                             long total_gt)
{
    if (scores_desc.empty() || total_gt <= 0) return 0.0;

    // cumulative TP/FP in score-desc order
    std::vector<long> cumTP(tp_flags_desc.size()), cumFP(tp_flags_desc.size());
    long tp=0, fp=0;
    for (size_t i=0;i<tp_flags_desc.size();++i){
        if (tp_flags_desc[i]) ++tp; else ++fp;
        cumTP[i]=tp; cumFP[i]=fp;
    }

    // precision/recall arrays
    std::vector<double> precis(tp_flags_desc.size()), recall(tp_flags_desc.size());
    for (size_t i=0;i<tp_flags_desc.size();++i){
        double p = (cumTP[i] + cumFP[i]) > 0 ? (double)cumTP[i] / (double)(cumTP[i]+cumFP[i]) : 0.0;
        double r = (double)cumTP[i] / (double)total_gt;
        precis[i]=p; recall[i]=r;
    }

    // Precision envelope (monotone non-increasing)
    for (int i=(int)precis.size()-2; i>=0; --i)
        if (precis[i] < precis[i+1]) precis[i] = precis[i+1];

    // Sample 101 recall points and average precision
    double ap = 0.0;
    for (int k=0; k<=100; ++k){
        double r = k / 100.0;
        double p_at_r = 0.0;
        for (size_t i=0;i<recall.size();++i){
            if (recall[i] >= r) { p_at_r = precis[i]; break; }
        }
        ap += p_at_r;
    }
    return ap / 101.0;
}

// ---------- MAIN ----------
int main(int argc, char* argv[]){
    if(argc < 5){
        std::cout << "Usage:\n  " << argv[0]
                  << " <model.onnx> <images_dir> <labels_dir> <out_dir>\n";
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string images_dir = argv[2];
    const std::string labels_dir = argv[3];
    const std::string out_dir    = argv[4];
    ensure_dir(out_dir);

    // ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8");
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    so.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    so.SetInterOpNumThreads(1);   
    so.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    Ort::Session session(env, model_path.c_str(), so);
    Ort::AllocatorWithDefaultOptions alloc;

    // IO names
    std::vector<std::string> input_names, output_names;
    for (size_t i=0; i<session.GetInputCount(); ++i) {
        Ort::AllocatedStringPtr n = session.GetInputNameAllocated(i, alloc);
        input_names.emplace_back(n.get());
    }
    for (size_t i=0; i<session.GetOutputCount(); ++i) {
        Ort::AllocatedStringPtr n = session.GetOutputNameAllocated(i, alloc);
        output_names.emplace_back(n.get());
    }
    std::vector<const char*> in_ptrs, out_ptrs;
    for (auto& s : input_names)  in_ptrs.push_back(s.c_str());
    for (auto& s : output_names) out_ptrs.push_back(s.c_str());

    // Images
    auto imgs = list_images(images_dir, 100000);
    if (imgs.empty()) {
        std::cerr << "No images found in: " << images_dir << "\n";
        return 1;
    }

    // CSV (detections + per-box info)
    const std::string csv_path = join_path(out_dir, "summary.csv");
    std::ofstream csv(csv_path.c_str());
    if (!csv) {
        std::cerr << "Failed to open CSV for writing: " << csv_path << "\n";
        return 1;
    }
    csv << "filename,idx,class_id,score,x,y,w,h,longest_px,longest_um,infer_ms\n";
    
    // Global metrics
    long total_gt_boxes = 0;
    std::vector<float> pr_scores_all; // AP buffers
    std::vector<int>   pr_tp_all;

    long TP = 0, FP = 0, FN = 0;
    double sumIoU_TP = 0.0;

    std::vector<double> times_ms;
    times_ms.reserve(imgs.size());
    bool model_single_class = false; // toggled if we detect class-agnostic outputs

    // ⬇️ Allocate once (not per loop)
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> ishape{1, 3, INPUT_H, INPUT_W};

    // Process images
    for (const std::string& img_path : imgs) {
        cv::Mat img_raw = cv::imread(img_path, cv::IMREAD_UNCHANGED);
        if (img_raw.empty()) {
            std::cerr << "Failed to read: " << img_path << "\n";
            continue;
        }

        cv::Mat img;
        if (img_raw.channels() == 1) 
            cv::cvtColor(img_raw, img, cv::COLOR_GRAY2BGR);  // grayscale → BGR
        else 
            img = img_raw;
            
        // Letterbox preprocess
        cv::Mat rgb640;
        LetterboxInfo L = letterbox_bgr_to_rgb_640(img, rgb640);

        // Fast path: do /255 and HWC->CHW in one go
        static std::vector<float> in_vals; // reuse
        cv::Mat blob = cv::dnn::blobFromImage(
            rgb640,
            1.0/255.0,                 // normalize here
            cv::Size(INPUT_W, INPUT_H),
            cv::Scalar(),
            /*swapRB=*/false,          // already RGB
            /*crop=*/false
        );
        // IMPORTANT: keep 'blob' alive until Run() returns
        Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
            mem,
            reinterpret_cast<float*>(blob.data),
            static_cast<size_t>(blob.total()),  // number of floats
            ishape.data(), ishape.size()
        );
        // in_vals.assign((float*)blob.datastart, (float*)blob.dataend);
    

        // std::vector<int64_t> ishape{1,3,INPUT_H,INPUT_W};
        // Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        // Ort::Value in_tensor = Ort::Value::CreateTensor<float>(mem, in_vals.data(), in_vals.size(),
        //                                                        ishape.data(), ishape.size());

        // Run inference (use first output)
        auto t0 = std::chrono::high_resolution_clock::now();
        auto outs = session.Run(Ort::RunOptions{nullptr}, in_ptrs.data(), &in_tensor, 1, out_ptrs.data(), 1);
        auto t1 = std::chrono::high_resolution_clock::now();
        double infer_ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
        times_ms.push_back(infer_ms);

        if(outs.empty() || !outs[0].IsTensor()){
            std::cerr<<"Unexpected output tensor for "<<img_path<<"\n";
            continue;
        }

        float* out = outs[0].GetTensorMutableData<float>();
        auto info  = outs[0].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> osh = info.GetShape();

        std::vector<Detection> dets;

        // ---- Case A: built-in NMS [N,6] or [1,N,6] (x1,y1,x2,y2,score,class) ----
        if ((osh.size()==2 && osh[1]==6) || (osh.size()==3 && osh[0]==1 && osh[2]==6)) {
            int64_t N = (osh.size()==2)? osh[0] : osh[1];
            const int S = 6;
            for (int64_t i=0;i<N;++i){
                float x1=out[i*S+0], y1=out[i*S+1], x2=out[i*S+2], y2=out[i*S+3];
                float score=out[i*S+4]; int cls=(int)out[i*S+5];
                if (score < CONF_THR) continue;
                // coords are in 640-letterboxed space -> map back
                undo_letterbox_xyxy(x1,y1,x2,y2,L,img.cols,img.rows);
                cv::Rect r(cv::Point2f(x1,y1), cv::Point2f(x2,y2));
                if(r.width<=0 || r.height<=0) continue;
                dets.push_back({r, score, cls});
            }
            if(!dets.empty()){
                int c0=dets[0].class_id; bool same=true;
                for(auto& d:dets) if(d.class_id!=c0){ same=false; break; }
                if(same) model_single_class = true;
            }
        } else {
            // ---- Case B: raw predictions ----
            if (osh.size() < 2) { std::cerr<<"Bad output rank\n"; continue; }
            int64_t d1 = (osh.size()>=2? osh[1] : 0);
            int64_t d2 = (osh.size()>=3? osh[2] : 0);

            bool chw_layout=false; // [1,C,N]
            int64_t N=0, C=0;

            // Prefer the dimension <= 8 as C (x,y,w,h,obj + classes)
            if (osh.size()==3) {
                if (d1 <= 8) { chw_layout=true;  C=d1; N=d2; }
                else if (d2 <= 8) { chw_layout=false; C=d2; N=d1; }
                else { // fallback
                    if (d1 > d2) { chw_layout=true; C=d1; N=d2; }
                    else         { chw_layout=false; C=d2; N=d1; }
                }
            } else { // [N,C]
                chw_layout=false; N=osh[0]; C=osh[1];
            }

            auto read = [&](int p,int k)->float{ return chw_layout? out[k*N + p] : out[p*C + k]; };

            if (C == 5) {
                // Class-agnostic: [xc, yc, w, h, score] in 640-letterboxed space
                model_single_class = true;
                for (int p=0; p<N; ++p) {
                    float xc=read(p,0), yc=read(p,1), w=read(p,2), h=read(p,3), score=read(p,4);
                    if (score < CONF_THR) continue;
                    float x1 = xc - 0.5f*w, y1 = yc - 0.5f*h, x2 = xc + 0.5f*w, y2 = yc + 0.5f*h;
                    undo_letterbox_xyxy(x1,y1,x2,y2,L,img.cols,img.rows);
                    cv::Rect r(cv::Point2f(x1,y1), cv::Point2f(x2,y2));
                    if (r.width<=0 || r.height<=0) continue;
                    dets.push_back({r, score, 0});
                }
            } else {
                // Multi-class: [xc,yc,w,h,obj,cls...]
                if (C < 6) { std::cerr<<"Bad pred dim C="<<C<<"\n"; continue; }
                int num_classes = (int)(C - 5);
                for (int p=0; p<N; ++p) {
                    float xc=read(p,0), yc=read(p,1), w=read(p,2), h=read(p,3), obj=read(p,4);
                    int best=-1; float bestcc=0.f;
                    for (int c=0;c<num_classes;++c){ float cc=read(p,5+c); if (cc>bestcc){ bestcc=cc; best=c; } }
                    float score = obj * bestcc;
                    if (score < CONF_THR) continue;
                    float x1 = xc - 0.5f*w, y1 = yc - 0.5f*h, x2 = xc + 0.5f*w, y2 = yc + 0.5f*h;
                    undo_letterbox_xyxy(x1,y1,x2,y2,L,img.cols,img.rows);
                    cv::Rect r(cv::Point2f(x1,y1), cv::Point2f(x2,y2));
                    if (r.width<=0 || r.height<=0) continue;
                    dets.push_back({r, score, best});
                }
            }

            // NMS since graph didn't do it
            std::vector<int> keep = NMS(dets, IOU_THR);
            std::vector<Detection> filtered; filtered.reserve(keep.size());
            for (int idx : keep) filtered.push_back(dets[idx]);
            dets.swap(filtered);
        }

        // Visualization + CSV
        cv::Mat vis = img.clone();
        int idx_counter = 0;
        for (const auto& d : dets){
            ++idx_counter;
            if(SAVE_VIS) cv::rectangle(vis, d.box, cv::Scalar(0,255,0), 2);
            int longest_px = std::max(d.box.width, d.box.height);
            double longest_um = longest_px * PX_TO_UM;
            if(SAVE_VIS){
                std::ostringstream label; label<<idx_counter<<": "<<longest_px<<"px / "<<std::fixed<<std::setprecision(2)<<longest_um<<"um";
                int base=0; cv::Size ts=cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
                int topY=std::max(0, d.box.y - ts.height - 6);
                cv::rectangle(vis, cv::Rect(d.box.x, topY, ts.width+6, ts.height+6), cv::Scalar(0,255,0), cv::FILLED);
                cv::putText(vis, label.str(), cv::Point(d.box.x+3, topY+ts.height+1), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
            }
            csv<<basename_no_ext(img_path)<<","<<idx_counter<<","<<d.class_id<<","<<std::fixed<<std::setprecision(4)<<d.score<<","
               <<d.box.x<<","<<d.box.y<<","<<d.box.width<<","<<d.box.height<<","<<longest_px<<","
               <<std::fixed<<std::setprecision(3)<<longest_um<<","<<std::fixed<<std::setprecision(3)<<infer_ms<<"\n";
        }
        if(SAVE_VIS){
            const std::string out_img = join_path(out_dir, basename_no_ext(img_path) + "_vis.jpg");
            if(!cv::imwrite(out_img, vis)){
                std::cerr<<"Failed to write: "<<out_img<<"\n";
            } else if (VERBOSE_PER_IMAGE){
                std::cout<<"[Saved] "<<out_img<<" ("<<dets.size()<<" dets, "<<std::fixed<<std::setprecision(2)<<infer_ms<<" ms)\n";
            }
        }

        // --------- ACCURACY + AP buffers ----------
        const std::string label_path = join_path(labels_dir, basename_no_ext(img_path) + ".txt");
        auto gts = load_gt_yolo(label_path, img.cols, img.rows);
        total_gt_boxes += (long)gts.size();

        // Decide class matching: if model is single-class OR GT has only one class id, ignore class mismatch
        bool gt_single_class = true;
        if(!gts.empty()){
            int c0 = gts[0].class_id;
            for(const auto& g: gts){ if(g.class_id != c0){ gt_single_class=false; break; } }
        }
        bool match_by_class = !(model_single_class || gt_single_class);

        // Evaluate and build PR flags
        PRAccum ev = evaluate_for_map(dets, gts, match_by_class);
        TP += ev.tp; FP += ev.fp; FN += ev.fn; sumIoU_TP += ev.sumIoU_TP;

        // Append to global PR buffers; we’ll re-sort globally later
        pr_scores_all.insert(pr_scores_all.end(), ev.scores.begin(), ev.scores.end());
        pr_tp_all.insert(pr_tp_all.end(),       ev.tp_flags.begin(), ev.tp_flags.end());
    }

    // --------- Final metrics ----------
    double precision = (TP + FP) > 0 ? (double)TP / (double)(TP + FP) : 0.0;
    double recall    = (TP + FN) > 0 ? (double)TP / (double)(TP + FN) : 0.0;
    double f1        = (precision + recall) > 0 ? 2.0 * precision * recall / (precision + recall) : 0.0;
    double avg_iou   = TP > 0 ? sumIoU_TP / (double)TP : 0.0;
    double avg_ms    = times_ms.empty()? 0.0 : std::accumulate(times_ms.begin(),times_ms.end(),0.0) / times_ms.size();

    // AP@0.50
    double ap50 = 0.0;
    if (COMPUTE_MAP && total_gt_boxes > 0 && !pr_scores_all.empty()) {
        // global sort by score desc
        std::vector<size_t> order(pr_scores_all.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t b){ return pr_scores_all[a] > pr_scores_all[b]; });

        std::vector<float> scores_sorted;   scores_sorted.reserve(order.size());
        std::vector<int>   tpflags_sorted;  tpflags_sorted.reserve(order.size());
        for (size_t i : order){ scores_sorted.push_back(pr_scores_all[i]); tpflags_sorted.push_back(pr_tp_all[i]); }

        ap50 = compute_ap_101(scores_sorted, tpflags_sorted, total_gt_boxes);
    }

    // 1) Write a tidy metrics.csv
    {
        const std::string metrics_path = join_path(out_dir, "metrics.csv");
        std::ofstream m(metrics_path.c_str());
        if (m) {
            std::time_t now = std::time(nullptr);
            char ts[64]; std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
            m << "timestamp,num_images,conf_thr,match_iou,tp,fp,fn,precision,recall,f1,avg_iou_tp,ap50,avg_infer_ms\n";
            m << ts << ","
              << imgs.size() << ","
              << std::fixed << std::setprecision(2) << CONF_THR << ","
              << std::fixed << std::setprecision(2) << MATCH_IOU << ","
              << TP << ","
              << FP << ","
              << FN << ","
              << std::fixed << std::setprecision(6) << precision << ","
              << std::fixed << std::setprecision(6) << recall << ","
              << std::fixed << std::setprecision(6) << f1 << ","
              << std::fixed << std::setprecision(6) << avg_iou << ","
              << std::fixed << std::setprecision(6) << ap50 << ","
              << std::fixed << std::setprecision(2) << avg_ms
              << "\n";
            m.close();
            std::cout << "Metrics written to: " << metrics_path << "\n";
        } else {
            std::cerr << "Failed to write metrics.csv\n";
        }
    }

    // 2) Append a clearly-labeled metrics section to summary.csv (with its own header)
    {
        std::ofstream s(csv_path.c_str(), std::ios::app);
        if (s) {
            s << "\n# METRICS\n";
            s << "precision,recall,f1,avg_iou_tp,ap50,avg_infer_ms,tp,fp,fn,num_images,conf_thr,match_iou\n";
            s << std::fixed
              << std::setprecision(6) << precision << ","
              << std::setprecision(6) << recall    << ","
              << std::setprecision(6) << f1        << ","
              << std::setprecision(6) << avg_iou   << ","
              << std::setprecision(6) << ap50      << ","
              << std::setprecision(2) << avg_ms    << ","
              << TP << "," << FP << "," << FN << ","
              << imgs.size() << ","
              << std::setprecision(2) << CONF_THR  << ","
              << std::setprecision(2) << MATCH_IOU << "\n";
            s.close();
        } else {
            std::cerr << "Failed to append metrics to summary.csv\n";
        }
    }

    // 3) Print to terminal (last)
    std::cout << "\n=== Evaluation @ conf>=" << CONF_THR << ", IoU>=" << MATCH_IOU << " ===\n";
    std::cout << "TP: " << TP << "  FP: " << FP << "  FN: " << FN << "\n";
    std::cout << std::fixed << std::setprecision(4)
              << "Precision: " << precision
              << "  Recall: "    << recall
              << "  F1: "        << f1
              << "  Avg IoU (TP): " << avg_iou
              << "  AP@0.50: " << ap50
              << "\n";
    std::cout << std::setprecision(2)
              << "Avg inference time: " << avg_ms << " ms\n";
    std::cout << "CSV summary written to: " << csv_path << "\n";

    return 0;
}

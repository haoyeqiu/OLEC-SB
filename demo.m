clc;clear;
addpath(genpath(pwd));
addpath('./utils/');
addpath('./metrics/');

rng(2026)
    
M = 20;
cntTimes = 10;

poolSize = 100;
bcIdx = initialRandom(poolSize, M, cntTimes);

rawFolder  = "./datasets";
procFolder = "./cmpResult/";

%% 1) Scan cmpResult: collect "existing dataset names" (= subfolder names)
procDirs = dir(fullfile(procFolder, "**", "*"));
procDirs = procDirs([procDirs.isdir]);

procDirNames = string({procDirs.name})';
procDirNames = procDirNames(procDirNames ~= "." & procDirNames ~= "..");
procNameSet  = unique(lower(strtrim(procDirNames)));

%% 2) Scan Raw: recursively find .mat files
% Specify the keep list of datasets
keep = lower(["Maternal Health Risk Data Set","Phishing",...
              "Students'_Dropout_and_Academic_Success","TUANDROMD",...
              "CNAE-9","Gender_Gap_in_Spanish_WP","HTRU2", ...
              "Crowdsourced_Mapping","COIL100","Steel_industry_data"]);
rawList = dir(fullfile(rawFolder,"**","*.mat"));
rawList = rawList(~[rawList.isdir]);

rawNames = lower(string(erase({rawList.name}, ".mat")));
rawList  = rawList(ismember(rawNames, keep));

%% 3) Check: if a subfolder with the same name already exists in cmpResult, skip; otherwise load and process
for i = 1:numel(rawList)
    rawPath = fullfile(rawList(i).folder, rawList(i).name);
    [~, dataName, ~] = fileparts(rawList(i).name);
    rawKey = lower(strtrim(string(dataName)));

    if ismember(rawKey, procNameSet)
        fprintf("[SKIP] Subfolder already exists: %s\n", dataName);
        continue;
    end

    fprintf("[DO]  Subfolder not found, start loading and processing: %s\n", dataName);

    % ======== Unified One-Step at Scale: Linear-Time Ensemble Clustering with Soft Balancing ========
    load(rawPath);
    
    n = length(gt);
    k = length(unique(gt));

    mean_result1 = [];
    std_result1 = [];
    para_lambda = [10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1];
    para_gamma = 1.1:0.1:3;
    nL = length(para_lambda);
    nG = length(para_gamma);
    allTimes    = cell(nL, nG);
    allResults  = cell(nL, nG);
    allavg_time = cell(nL, nG);
    allstd_time = cell(nL, nG);
    allobjs = cell(nL, nG);
    allalphas = cell(nL, nG);
    allclusterings = cell(nL, nG);
    for lambda_idx = 1:length(para_lambda)
        for gamma_idx = 1:length(para_gamma)
            lambda = para_lambda(lambda_idx);
            gamma = para_gamma(gamma_idx);
            resultsAll = cell(1, cntTimes);
            times = zeros(1, cntTimes);
            objsAll = cell(1, cntTimes);
            alphasAll = cell(1, cntTimes);
            clusteringsAll = cell(1, cntTimes);
            parfor idx = 1:cntTimes
                clusterings = members(:, bcIdx(idx, :));
                tStart = tic;
                [result1, obj, alpha] = main(clusterings, M, k, lambda, gamma);
                times(idx) = toc(tStart);
                resultsAll{idx} = result1;
                objsAll{idx} = obj;
                alphasAll{idx} = alpha;
                clusteringsAll{idx} = clusterings;
                measure1(:, idx) = computeMetrics(result1, gt);
            end
    
            avg_time = mean(times);
            std_time = std(times);
    
            allTimes{lambda_idx, gamma_idx}   = times;
            allResults{lambda_idx, gamma_idx} = resultsAll;
            allavg_time{lambda_idx, gamma_idx} = avg_time;
            allstd_time{lambda_idx, gamma_idx} = std_time;
            allobjs{lambda_idx, gamma_idx} = objsAll;
            allalphas{lambda_idx, gamma_idx} = alphasAll;
            allclusterings{lambda_idx, gamma_idx} = clusteringsAll;
        
            mean_measure1 = mean(measure1, 2)';
            std_measure1 = std(measure1, 0, 2)';
            mean_result1 = [mean_result1; [lambda_idx, gamma, mean_measure1]];
            std_result1 = [std_result1; [lambda_idx, gamma, std_measure1]];
        
            fprintf("lambda:%.3f,gamma:%.3f,NMI:%.4f,ARI:%.4f,F1:%.4f,Purity:%.4f,ACC:%.4f,Kappa:%.4f,Precision:%.4f,Recall:%.4f\n",lambda,gamma,mean_measure1);
        end
    end

    [~, Idx] = max(prod(mean_result1(:,end-7:end),2));
    mean1 = mean_result1(Idx,:);
    std1 = std_result1(Idx,:);

    if (exist(['./cmpResult/', dataName], 'dir') == 0)
        mkdir(['./cmpResult/', dataName])
    end
    if (exist(['./para/', dataName], 'dir') == 0)
        mkdir(['./para/', dataName])
    end

    save(['./cmpResult/', dataName, '/mean1'], "mean1");
    save(['./cmpResult/', dataName, '/std1'], "std1");

    save(['./para/', dataName, '/mean_result1'], "mean_result1");
    save(['./para/', dataName, '/std_result1'], "std_result1");

    outDir = fullfile('.', 'results', dataName);
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    
    outFile = fullfile(outDir, 'data_save1.mat');
    save(outFile, "allTimes", "allResults", "allavg_time", "allstd_time", "allobjs", "allalphas", "allclusterings");

end
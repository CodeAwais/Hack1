"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Brain,
  Upload,
  Loader2,
  AlertCircle,
} from "lucide-react";

export default function Home() {
  // State for patient information
  const [patientInfo, setPatientInfo] = useState({
    age: "40",
    sex: "female",
    symptoms: "",
    family_history: "yes",
    weight: "60",
    region: "North America",
    smoking_history: "no",
    ebv: "yes",
  });

  // Available symptoms
  const availableSymptoms = [
    "Vision problems (blurry vision, optic neuritis)",
    "Numbness or tingling",
    "Muscle weakness or spasms",
    "Fatigue",
    "Balance & coordination issues",
    "Cognitive changes",
    "Bladder or bowel issues"
  ];

  // State for selected symptoms
  const [selectedSymptoms, setSelectedSymptoms] = useState<string[]>([]);

  // State for file upload
  const [file, setFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [diagnosis, setDiagnosis] = useState<{
    diagnosis?: string;
    confidence?: string;
    summary?: string;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Handle input changes
  const handleInputChange = (field: string, value: string) => {
    setPatientInfo((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  // Handle symptom selection
  const handleSymptomChange = (symptom: string, checked: boolean) => {
    if (checked) {
      const newSymptoms = [...selectedSymptoms, symptom];
      setSelectedSymptoms(newSymptoms);
      setPatientInfo(prev => ({
        ...prev,
        symptoms: newSymptoms.join(", ")
      }));
    } else {
      const newSymptoms = selectedSymptoms.filter(s => s !== symptom);
      setSelectedSymptoms(newSymptoms);
      setPatientInfo(prev => ({
        ...prev,
        symptoms: newSymptoms.join(", ")
      }));
    }
  };

  // Handle file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files ? e.target.files[0] : null;
    if (selectedFile) {
      setFile(selectedFile);
      setFileName(selectedFile.name);
      setError(null);
    }
  };

  // Handle drag and drop
  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      setFile(droppedFile);
      setFileName(droppedFile.name);
      setError(null);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  // Submit data to the API
  const handleSubmit = async () => {
    if (!file) {
      setError("Please upload an MRI scan");
      return;
    }

    setIsLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append("file", file);
      
      // Add all patient info to the form data
      Object.entries(patientInfo).forEach(([key, value]) => {
        formData.append(key, value);
      });

      const response = await fetch("http://localhost:8000/api/perplexity/", {
        method: "POST",
        body: formData,
      });

      if (response.status === 400) {
        throw new Error("Only DICOM (.dcm) and DICOM files are supported");
      }

      if (!response.ok) {
        throw new Error(`HTTP error status: ${response.status}`);
      }

      const result = await response.json();
      setDiagnosis(result);
    } catch (err) {
      console.error("Error submitting data:", err);
      setError(err instanceof Error ? err.message : "Failed to get diagnosis. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  // Function to get diagnosis status color
  const getDiagnosisColor = (diagnosis: string | undefined) => {
    if (!diagnosis) return "bg-gray-100 text-gray-700";
    
    const lowerDiagnosis = diagnosis.toLowerCase();
    if (lowerDiagnosis.includes("multiple sclerosis") || lowerDiagnosis.includes("likely ms")) {
      return "bg-red-100 text-red-700";
    } else if (lowerDiagnosis.includes("healthy") || lowerDiagnosis.includes("no ms")) {
      return "bg-green-100 text-green-700";
    } else if (lowerDiagnosis.includes("uncertain") || lowerDiagnosis.includes("possible")) {
      return "bg-amber-100 text-amber-700";
    } else {
      return "bg-blue-100 text-blue-700";
    }
  };

  // Format confidence display
  // const formatConfidence = (confidence: string | undefined) => {
  //   if (!confidence) return "";
    
  //   // Remove any markdown formatting that might be present
  //   const cleanedText = confidence
  //     .replace(/\*\*/g, "")
  //     .replace(/\*/g, "")
  //     .replace(/#/g, "")
  //     .replace(/```/g, "")
  //     .replace(/\n+/g, " ")
  //     .trim();
    
  //   return cleanedText;
  // };

  // Format summary display
  const formatSummary = (summary: string | undefined) => {
    if (!summary) return "";
    
    // Remove any markdown formatting that might be present
    const cleanedText = summary
      .replace(/\*\*/g, "")
      .replace(/\*/g, "")
      .replace(/#/g, "")
      .replace(/```/g, "")
      .replace(/\n+/g, " ")
      .trim();
    
    return cleanedText;
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-slate-50">
      <header className="flex items-center justify-between bg-white p-5 shadow-md">
        <div className="flex items-center gap-4">
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-600">
            <Brain className="h-7 w-7 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-blue-800">MSDetect</h1>
            <p className="text-sm text-slate-600">
              AI-Powered Multiple Sclerosis Screening Tool
            </p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto p-6">
        <div className="grid gap-8 md:grid-cols-2">
          {/* Patient Information Panel */}
          <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-lg">
            <h2 className="mb-3 text-xl font-semibold text-slate-800">Patient Profile</h2>
            <p className="mb-6 text-sm text-slate-500">
              Complete all fields below and upload an MRI scan for analysis.
            </p>

            <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
              <div className="space-y-2">
                <label htmlFor="age" className="text-sm font-medium text-slate-700">
                  Patient Age
                </label>
                <Input 
                  id="age" 
                  value={patientInfo.age} 
                  onChange={(e) => handleInputChange("age", e.target.value)} 
                  className="border-slate-300 focus:border-blue-500"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-slate-700">Biological Sex</label>
                <Select 
                  value={patientInfo.sex} 
                  onValueChange={(value) => handleInputChange("sex", value)}
                >
                  <SelectTrigger className="border-slate-300 focus:border-blue-500">
                    <SelectValue placeholder="Select sex" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="male">Male</SelectItem>
                    <SelectItem value="female">Female</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-slate-700">
                  Family History of MS
                </label>
                <Select 
                  value={patientInfo.family_history} 
                  onValueChange={(value) => handleInputChange("family_history", value)}
                >
                  <SelectTrigger className="border-slate-300 focus:border-blue-500">
                    <SelectValue placeholder="Yes/No" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="yes">Yes</SelectItem>
                    <SelectItem value="no">No</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <label htmlFor="weight" className="text-sm font-medium text-slate-700">
                  Weight (kg)
                </label>
                <Input 
                  id="weight" 
                  value={patientInfo.weight} 
                  onChange={(e) => handleInputChange("weight", e.target.value)}
                  className="border-slate-300 focus:border-blue-500"
                />
              </div>
              <div className="space-y-2">
                <label htmlFor="region" className="text-sm font-medium text-slate-700">
                  Geographic Region
                </label>
                <Input 
                  id="region" 
                  value={patientInfo.region} 
                  onChange={(e) => handleInputChange("region", e.target.value)}
                  className="border-slate-300 focus:border-blue-500"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-slate-700">Smoking History</label>
                <Select 
                  value={patientInfo.smoking_history} 
                  onValueChange={(value) => handleInputChange("smoking_history", value)}
                >
                  <SelectTrigger className="border-slate-300 focus:border-blue-500">
                    <SelectValue placeholder="Select" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="yes">Yes</SelectItem>
                    <SelectItem value="no">No</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-slate-700">
                  History of Epstein-Barr Virus
                </label>
                <Select 
                  value={patientInfo.ebv} 
                  onValueChange={(value) => handleInputChange("ebv", value)}
                >
                  <SelectTrigger className="border-slate-300 focus:border-blue-500">
                    <SelectValue placeholder="Select" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="yes">Yes</SelectItem>
                    <SelectItem value="no">No</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Symptoms Checklist */}
            <div className="mt-6">
              <label className="text-sm font-medium text-slate-700">
                Symptoms (select all that apply)
              </label>
              <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-2">
                {availableSymptoms.map((symptom) => (
                  <div key={symptom} className="flex items-center space-x-2">
                    <Checkbox 
                      id={`symptom-${symptom}`} 
                      checked={selectedSymptoms.includes(symptom)}
                      onCheckedChange={(checked) => 
                        handleSymptomChange(symptom, checked === true)
                      }
                      className="border-slate-300 data-[state=checked]:bg-blue-600"
                    />
                    <label 
                      htmlFor={`symptom-${symptom}`} 
                      className="text-sm text-slate-700"
                    >
                      {symptom}
                    </label>
                  </div>
                ))}
              </div>
            </div>

            {/* File Upload Section */}
            <div className="mt-6 space-y-2">
              <label className="text-sm font-medium text-slate-700">MRI Scan Upload</label>
              <div 
                className={`flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-6 transition-colors ${
                  file ? "border-blue-400 bg-blue-50" : "border-slate-300 hover:border-blue-300"
                }`}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
              >
                <input
                  type="file"
                  id="file-upload"
                  className="hidden"
                  onChange={handleFileChange}
                  accept=".dcm,.dicom"
                />
                <label htmlFor="file-upload" className="cursor-pointer">
                  {file ? (
                    <>
                      <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-blue-100 text-blue-600">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      </div>
                      <p className="mb-1 text-sm font-medium text-blue-600">{fileName}</p>
                      <p className="text-xs text-slate-500">Click to select a different file</p>
                    </>
                  ) : (
                    <>
                      <Upload className="mb-3 h-10 w-10 text-slate-400" />
                      <p className="mb-2 text-sm font-medium text-slate-600">
                        Drop your MRI scan here or click to browse
                      </p>
                      <p className="text-xs text-slate-500">
                        Supported formats: .dicom and .dcm files only
                      </p>
                    </>
                  )}
                </label>
              </div>
              {error && (
                <div className="mt-2 flex items-center text-sm text-red-600">
                  <AlertCircle className="mr-1 h-4 w-4" />
                  {error}
                </div>
              )}
            </div>
            <Button 
              className="mt-6 w-full bg-blue-600 text-white hover:bg-blue-700"
              onClick={handleSubmit}
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Analyzing MRI Data...
                </>
              ) : (
                "Generate MS Risk Assessment"
              )}
            </Button>
          </div>

          {/* Diagnosis Results Panel */}
          <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-lg">
            <h2 className="mb-3 text-xl font-semibold text-slate-800">Clinical Assessment</h2>
            <p className="mb-6 text-sm text-slate-500">
              AI-powered analysis and diagnostic information
            </p>

            {diagnosis ? (
              <div className="space-y-6">
                <div className="rounded-xl bg-blue-50 p-5">
                  <h3 className="mb-3 text-lg font-semibold text-blue-800">Diagnosis</h3>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-slate-700">Assessment:</span>
                    <span className={`rounded-full px-4 py-1.5 text-sm font-semibold ${getDiagnosisColor(diagnosis.diagnosis)}`}>
                      {diagnosis.diagnosis || "Unknown"}
                    </span>
                  </div>
                </div>

                <div className="space-y-3">
                  <h3 className="text-lg font-semibold text-slate-800">MRI Analysis</h3>
                  <div className="rounded-xl bg-slate-50 p-5">
                    <p className="text-sm leading-relaxed text-slate-700">
                      {diagnosis.confidence}%
                    </p>
                  </div>
                </div>

                <div className="space-y-3">
                  <h3 className="text-lg font-semibold text-slate-800">Recommendations</h3>
                  <div className="rounded-xl bg-slate-50 p-5">
                    <p className="text-sm leading-relaxed text-slate-700">
                      {formatSummary(diagnosis.summary)}
                    </p>
                  </div>
                </div>

                <div className="mt-6 flex items-center text-xs text-slate-500">
                  <span>Assessment generated: {new Date().toLocaleString()}</span>
                </div>
              </div>
            ) : (
              <div className="flex h-[400px] flex-col items-center justify-center">
                <div className="mb-6 flex h-24 w-24 items-center justify-center rounded-full border-2 border-blue-100 bg-blue-50">
                  <Brain className="h-14 w-14 text-blue-300" />
                </div>
                <h3 className="mb-3 text-lg font-semibold text-slate-700">Waiting for Assessment</h3>
                <p className="max-w-md text-center text-sm leading-relaxed text-slate-500">
                  Complete the patient form and upload an MRI scan to receive an AI-assisted multiple sclerosis risk assessment.
                </p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
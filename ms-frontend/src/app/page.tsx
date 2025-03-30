"use client";

import { useState } from "react";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  MessageCircleQuestionIcon as CircleQuestion,
  Upload,
  Loader2,
} from "lucide-react";

export default function Home() {
  // State for patient information
  const [patientInfo, setPatientInfo] = useState({
    age: "40",
    sex: "male",
    relative: "no",
    weight: "60",
    region: "North America",
    smoking: "yes",
    ebv: "yes",
  });

  // State for file upload
  const [file, setFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [diagnosis, setDiagnosis] = useState<{
    riskLevel?: string;
    report?: string;
    recommendations?: string[];
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Handle input changes
  const handleInputChange = (field: string, value: string) => {
    setPatientInfo((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  // Handle file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files ? e.target.files[0] : null;
    if (selectedFile) {
      setFile(selectedFile);
      setFileName(selectedFile.name);
    }
  };

  // Handle drag and drop
  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      setFile(droppedFile);
      setFileName(droppedFile.name);
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

      const response = await fetch("http://localhost:8000/api/perplexity", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const result = await response.json();
      setDiagnosis(result);
    } catch (err) {
      console.error("Error submitting data:", err);
      setError("Failed to get diagnosis. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="flex items-center justify-between bg-white p-4 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-md bg-slate-900">
            <Image
              src="/placeholder.svg?height=24&width=24"
              width={24}
              height={24}
              alt=""
              className="invert"
            />
          </div>
          <div>
            <h1 className="text-lg font-semibold">MSDetect</h1>
            <p className="text-xs text-muted-foreground">
              Early Detection of Multiple Sclerosis (MS)
            </p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto grid gap-6 p-6 md:grid-cols-2">
        {/* Patient Information Panel */}
        <div className="rounded-lg border bg-white p-6 shadow-sm">
          <h2 className="mb-2 text-lg font-medium">Patient Information</h2>
          <p className="mb-6 text-sm text-muted-foreground">
            Enter patient details and imaging information to generate a
            diagnosis.
          </p>

          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-2">
              <label htmlFor="age" className="text-sm font-medium">
                Age
              </label>
              <Input 
                id="age" 
                value={patientInfo.age} 
                onChange={(e) => handleInputChange("age", e.target.value)} 
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Sex</label>
              <Select 
                value={patientInfo.sex} 
                onValueChange={(value) => handleInputChange("sex", value)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select sex" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="male">Male</SelectItem>
                  <SelectItem value="female">Female</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">
                Do you have a relative with multiple sclerosis?
              </label>
              <Select 
                value={patientInfo.relative} 
                onValueChange={(value) => handleInputChange("relative", value)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Yes/No" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="yes">Yes</SelectItem>
                  <SelectItem value="no">No</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <label htmlFor="weight" className="text-sm font-medium">
                Weight (kg)
              </label>
              <Input 
                id="weight" 
                value={patientInfo.weight} 
                onChange={(e) => handleInputChange("weight", e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <label htmlFor="region" className="text-sm font-medium">
                Region
              </label>
              <Input 
                id="region" 
                value={patientInfo.region} 
                onChange={(e) => handleInputChange("region", e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Smoking History</label>
              <Select 
                value={patientInfo.smoking} 
                onValueChange={(value) => handleInputChange("smoking", value)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="yes">Yes</SelectItem>
                  <SelectItem value="no">No</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">
                Infected with Epstein-Barr Virus (EBV) Before?
              </label>
              <Select 
                value={patientInfo.ebv} 
                onValueChange={(value) => handleInputChange("ebv", value)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="yes">Yes</SelectItem>
                  <SelectItem value="no">No</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* File Upload Section */}
          <div className="mt-6 space-y-2">
            <label className="text-sm font-medium">Upload MRI Scan</label>
            <div 
              className={`flex flex-col items-center justify-center rounded-md border-2 border-dashed p-6 ${
                file ? "border-green-500 bg-green-50" : ""
              }`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
            >
              <input
                type="file"
                id="file-upload"
                className="hidden"
                onChange={handleFileChange}
                accept=".dcm,.jpg,.jpeg,.png"
              />
              <label htmlFor="file-upload" className="cursor-pointer">
                {file ? (
                  <>
                    <div className="mb-2 flex items-center justify-center">
                      <div className="flex h-8 w-8 items-center justify-center rounded-full bg-green-100 text-green-600">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      </div>
                    </div>
                    <p className="mb-1 text-sm font-medium text-green-600">{fileName}</p>
                    <p className="text-xs text-muted-foreground">Click to change file</p>
                  </>
                ) : (
                  <>
                    <Upload className="mb-2 h-8 w-8 text-gray-400" />
                    <p className="mb-2 text-sm text-muted-foreground">
                      Drag and drop or click to upload
                    </p>
                    <p className="mb-4 text-xs text-muted-foreground">
                      Supported formats: DICOM, JPG, PNG
                    </p>
                  </>
                )}
              </label>
            </div>
            {error && <p className="mt-2 text-sm text-red-500">{error}</p>}
          </div>
          <Button 
            className="mt-6 w-full bg-slate-900 text-white hover:bg-slate-800"
            onClick={handleSubmit}
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Processing...
              </>
            ) : (
              "Generate Diagnosis"
            )}
          </Button>
        </div>

        {/* Diagnosis Results Panel */}
        <div className="rounded-lg border bg-white p-6 shadow-sm">
          <h2 className="mb-2 text-lg font-medium">Diagnosis Results</h2>
          <p className="mb-6 text-sm text-muted-foreground">
            AI-assisted analysis of patient data
          </p>

          {diagnosis ? (
            <div className="space-y-4">
              <div className="rounded-md bg-slate-50 p-4">
                <h3 className="mb-2 text-md font-medium">MS Risk Assessment</h3>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Risk Level:</span>
                  <span className={`rounded-full px-3 py-1 text-xs font-semibold ${
                    diagnosis.riskLevel === "High" 
                      ? "bg-red-100 text-red-700" 
                      : diagnosis.riskLevel === "Medium" 
                        ? "bg-yellow-100 text-yellow-700" 
                        : "bg-green-100 text-green-700"
                  }`}>
                    {diagnosis.riskLevel || "Unknown"}
                  </span>
                </div>
              </div>

              <div className="space-y-3">
                <h3 className="text-md font-medium">Analysis Report</h3>
                <div className="rounded-md bg-slate-50 p-4">
                  <p className="text-sm">{diagnosis.report || "No detailed report available."}</p>
                </div>
              </div>

              <div className="space-y-3">
                <h3 className="text-md font-medium">Recommendations</h3>
                <div className="rounded-md bg-slate-50 p-4">
                  <ul className="space-y-2 text-sm">
                    {diagnosis.recommendations ? (
                      diagnosis.recommendations.map((rec, idx) => (
                        <li key={idx} className="flex items-start">
                          <span className="mr-2 text-slate-600">•</span>
                          <span>{rec}</span>
                        </li>
                      ))
                    ) : (
                      <li>No recommendations available.</li>
                    )}
                  </ul>
                </div>
              </div>

              <div className="mt-4 text-xs text-muted-foreground">
                Generated on {new Date().toLocaleString()}
              </div>
            </div>
          ) : (
            <div className="flex h-[300px] flex-col items-center justify-center">
              <div className="mb-4 flex h-20 w-20 items-center justify-center rounded-full border-2">
                <CircleQuestion className="h-12 w-12 text-gray-400" />
              </div>
              <h3 className="mb-2 text-lg font-medium">No diagnosis yet</h3>
              <p className="text-center text-sm text-muted-foreground">
                Complete the patient information form and upload an MRI scan to generate a diagnosis.
              </p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
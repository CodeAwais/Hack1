import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { MessageCircleQuestionIcon as CircleQuestion, Upload } from "lucide-react"

export default function Home() {
  return (
    <div className="min-h-screen bg-slate-50">
      <header className="flex items-center justify-between bg-white p-4 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-md bg-slate-900">
            <Image src="/placeholder.svg?height=24&width=24" width={24} height={24} alt="" className="invert" />
          </div>
          <div>
            <h1 className="text-lg font-semibold">MSDetect</h1>
            <p className="text-xs text-muted-foreground">Early Detection of Multiple Sclerosis (MS)</p>
          </div>
        </div>
        {/* <div className="flex items-center gap-4">
          <span className="text-sm">Welcome, Dr. Radwan</span>
        </div> */}
      </header>

      {/* Main Content */}
      <main className="container mx-auto grid gap-6 p-6 md:grid-cols-2">
        {/* Patient Information Panel */}
        <div className="rounded-lg border bg-white p-6 shadow-sm">
          <h2 className="mb-2 text-lg font-medium">Patient Information</h2>
          <p className="mb-6 text-sm text-muted-foreground">
            Enter patient details and imaging information to generate a diagnosis.
          </p>

          <Tabs defaultValue="demographics">
            <TabsList className="grid w-full">
              <TabsTrigger value="demographics">Demographics</TabsTrigger>
            </TabsList>
            <TabsContent value="demographics" className="mt-4 space-y-6">
              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-2">
                  <label htmlFor="age" className="text-sm font-medium">
                    Age
                  </label>
                  <Input id="age" defaultValue="40" />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Sex</label>
                  <Select defaultValue="male">
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
                  <label className="text-sm font-medium">Do you have a relative with multiple sclerosis?</label>
                  <Select defaultValue="no">
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
                  <Input id="weight" defaultValue="60" />
                </div>
                <div className="space-y-2">
                  <label htmlFor="region" className="text-sm font-medium">
                    Region
                  </label>
                  <Input id="region" defaultValue="North America" />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Smoking History</label>
                  <Select defaultValue="yes">
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
                  <label className="text-sm font-medium">Infected with Epstein-Barr Virus (EBV) Before?</label>
                  <Select defaultValue="yes">
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

              <div className="mt-6 space-y-2">
                <label className="text-sm font-medium">Upload MRI Scan</label>
                <div className="flex flex-col items-center justify-center rounded-md border-2 border-dashed p-6">
                  <Upload className="mb-2 h-8 w-8 text-gray-400" />
                  <p className="mb-2 text-sm text-muted-foreground">Drag and drop or click to upload</p>
                  <p className="mb-4 text-xs text-muted-foreground">Supported formats: DICOM, JPG, PNG</p>
                  <div className="relative">
                    <label htmlFor="file-upload" className="relative z-10 cursor-pointer">
                      <Button variant="outline" size="sm">Select File</Button>
                    </label>
                    <Input
                      type="file"
                      id="file-upload"
                      className="absolute inset-0 w-full h-full cursor-pointer opacity-0"
                      accept=".dcm,.jpg,.jpeg,.png"
                    />
                  </div>
                </div>
              </div>
            </TabsContent>
          </Tabs>

          <Button className="mt-6 w-full bg-slate-900 text-white hover:bg-slate-800">Generate Diagnosis</Button>
        </div>

        {/* Diagnosis Results Panel */}
        <div className="rounded-lg border bg-white p-6 shadow-sm">
          <h2 className="mb-2 text-lg font-medium">Diagnosis Results</h2>
          <p className="mb-6 text-sm text-muted-foreground">AI-assisted analysis of patient data</p>

          <div className="flex h-[300px] flex-col items-center justify-center">
            <div className="mb-4 flex h-20 w-20 items-center justify-center rounded-full border-2">
              <CircleQuestion className="h-12 w-12 text-gray-400" />
            </div>
            <h3 className="mb-2 text-lg font-medium">No diagnosis yet</h3>
            <p className="text-center text-sm text-muted-foreground">
              Complete the patient information form to generate a diagnosis.
            </p>
          </div>
        </div>
      </main>
    </div>
  )
}


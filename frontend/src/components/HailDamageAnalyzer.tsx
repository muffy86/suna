import React, { useState, useRef, useCallback } from 'react';
import { Upload, Camera, FileImage, AlertCircle, CheckCircle, DollarSign, Clock, Car, MapPin } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';

interface DentDetection {
  location: string;
  size: string;
  severity: 'minor' | 'moderate' | 'severe';
  panel: string;
  estimated_cost: number;
  confidence: number;
}

interface DamageAssessment {
  total_dents: number;
  dents_by_severity: {
    minor: number;
    moderate: number;
    severe: number;
  };
  total_estimated_cost: number;
  repair_method: string;
  is_total_loss: boolean;
  confidence_score: number;
}

interface AnalysisResult {
  assessment: DamageAssessment;
  detailed_report: string;
  processed_image: string;
  detected_dents: DentDetection[];
}

const HailDamageAnalyzer: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [vehicleInfo, setVehicleInfo] = useState({
    year: '',
    make: '',
    model: '',
    value: ''
  });
  const [analysisMode, setAnalysisMode] = useState<'quick' | 'detailed' | 'insurance'>('detailed');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = useCallback((file: File) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setError(null);
      setAnalysisResult(null);
    } else {
      setError('Please select a valid image file');
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  }, [handleFileSelect]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const analyzeImage = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    try {
      // Create FormData for file upload
      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('analysis_mode', analysisMode);
      
      if (vehicleInfo.year) formData.append('vehicle_year', vehicleInfo.year);
      if (vehicleInfo.make) formData.append('vehicle_make', vehicleInfo.make);
      if (vehicleInfo.model) formData.append('vehicle_model', vehicleInfo.model);
      if (vehicleInfo.value) formData.append('vehicle_value', vehicleInfo.value);

      // Call the backend API
      const response = await fetch('/api/analyze-hail-damage', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const result = await response.json();
      setAnalysisResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'minor': return 'bg-green-100 text-green-800';
      case 'moderate': return 'bg-yellow-100 text-yellow-800';
      case 'severe': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getRepairMethodColor = (method: string) => {
    if (method.includes('PDR')) return 'bg-green-100 text-green-800';
    if (method.includes('Traditional')) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold text-gray-900">AI Hail Damage Analyzer</h1>
        <p className="text-gray-600">Professional vehicle damage assessment for the PDR industry</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Camera className="h-5 w-5" />
              Upload Vehicle Image
            </CardTitle>
            <CardDescription>
              Upload a clear image of the vehicle showing hail damage
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div
              className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-gray-400 transition-colors cursor-pointer"
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onClick={() => fileInputRef.current?.click()}
            >
              {previewUrl ? (
                <div className="space-y-4">
                  <img
                    src={previewUrl}
                    alt="Vehicle preview"
                    className="max-h-64 mx-auto rounded-lg shadow-md"
                  />
                  <p className="text-sm text-gray-600">{selectedFile?.name}</p>
                </div>
              ) : (
                <div className="space-y-4">
                  <FileImage className="h-12 w-12 mx-auto text-gray-400" />
                  <div>
                    <p className="text-lg font-medium text-gray-900">Drop image here or click to upload</p>
                    <p className="text-sm text-gray-600">Supports JPG, PNG, WebP formats</p>
                  </div>
                </div>
              )}
            </div>
            
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
              className="hidden"
            />

            {/* Vehicle Information */}
            <div className="space-y-4">
              <h3 className="font-medium text-gray-900">Vehicle Information (Optional)</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="year">Year</Label>
                  <Input
                    id="year"
                    type="number"
                    placeholder="2020"
                    value={vehicleInfo.year}
                    onChange={(e) => setVehicleInfo(prev => ({ ...prev, year: e.target.value }))}
                  />
                </div>
                <div>
                  <Label htmlFor="make">Make</Label>
                  <Input
                    id="make"
                    placeholder="Toyota"
                    value={vehicleInfo.make}
                    onChange={(e) => setVehicleInfo(prev => ({ ...prev, make: e.target.value }))}
                  />
                </div>
                <div>
                  <Label htmlFor="model">Model</Label>
                  <Input
                    id="model"
                    placeholder="Camry"
                    value={vehicleInfo.model}
                    onChange={(e) => setVehicleInfo(prev => ({ ...prev, model: e.target.value }))}
                  />
                </div>
                <div>
                  <Label htmlFor="value">Value ($)</Label>
                  <Input
                    id="value"
                    type="number"
                    placeholder="25000"
                    value={vehicleInfo.value}
                    onChange={(e) => setVehicleInfo(prev => ({ ...prev, value: e.target.value }))}
                  />
                </div>
              </div>
            </div>

            {/* Analysis Mode */}
            <div className="space-y-2">
              <Label>Analysis Mode</Label>
              <Select value={analysisMode} onValueChange={(value: any) => setAnalysisMode(value)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="quick">Quick Scan</SelectItem>
                  <SelectItem value="detailed">Detailed Analysis</SelectItem>
                  <SelectItem value="insurance">Insurance Report</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <Button
              onClick={analyzeImage}
              disabled={!selectedFile || isAnalyzing}
              className="w-full"
              size="lg"
            >
              {isAnalyzing ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                  Analyzing...
                </>
              ) : (
                'Analyze Hail Damage'
              )}
            </Button>

            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        {/* Results Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle className="h-5 w-5" />
              Analysis Results
            </CardTitle>
          </CardHeader>
          <CardContent>
            {!analysisResult ? (
              <div className="text-center py-12 text-gray-500">
                <AlertCircle className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Upload an image and click analyze to see results</p>
              </div>
            ) : (
              <Tabs defaultValue="summary" className="space-y-4">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="summary">Summary</TabsTrigger>
                  <TabsTrigger value="details">Details</TabsTrigger>
                  <TabsTrigger value="image">Processed</TabsTrigger>
                </TabsList>

                <TabsContent value="summary" className="space-y-4">
                  {/* Key Metrics */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-blue-50 p-4 rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <MapPin className="h-4 w-4 text-blue-600" />
                        <span className="text-sm font-medium text-blue-900">Total Dents</span>
                      </div>
                      <p className="text-2xl font-bold text-blue-900">{analysisResult.assessment.total_dents}</p>
                    </div>
                    <div className="bg-green-50 p-4 rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <DollarSign className="h-4 w-4 text-green-600" />
                        <span className="text-sm font-medium text-green-900">Est. Cost</span>
                      </div>
                      <p className="text-2xl font-bold text-green-900">
                        ${analysisResult.assessment.total_estimated_cost.toLocaleString()}
                      </p>
                    </div>
                  </div>

                  {/* Severity Breakdown */}
                  <div className="space-y-3">
                    <h4 className="font-medium text-gray-900">Damage Severity</h4>
                    <div className="space-y-2">
                      {Object.entries(analysisResult.assessment.dents_by_severity).map(([severity, count]) => (
                        <div key={severity} className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Badge className={getSeverityColor(severity)}>
                              {severity.charAt(0).toUpperCase() + severity.slice(1)}
                            </Badge>
                            <span className="text-sm text-gray-600">{count} dents</span>
                          </div>
                          <div className="w-24">
                            <Progress 
                              value={(count / analysisResult.assessment.total_dents) * 100} 
                              className="h-2"
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Repair Method */}
                  <div className="space-y-2">
                    <h4 className="font-medium text-gray-900">Recommended Repair</h4>
                    <Badge className={getRepairMethodColor(analysisResult.assessment.repair_method)} variant="outline">
                      {analysisResult.assessment.repair_method}
                    </Badge>
                  </div>

                  {/* Confidence Score */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <h4 className="font-medium text-gray-900">Analysis Confidence</h4>
                      <span className="text-sm font-medium">
                        {(analysisResult.assessment.confidence_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <Progress value={analysisResult.assessment.confidence_score * 100} className="h-2" />
                  </div>

                  {/* Total Loss Warning */}
                  {analysisResult.assessment.is_total_loss && (
                    <Alert variant="destructive">
                      <AlertCircle className="h-4 w-4" />
                      <AlertDescription>
                        <strong>Total Loss:</strong> Repair costs exceed 75% of vehicle value
                      </AlertDescription>
                    </Alert>
                  )}
                </TabsContent>

                <TabsContent value="details" className="space-y-4">
                  <div className="space-y-4">
                    <h4 className="font-medium text-gray-900">Detected Dents ({analysisResult.detected_dents.length})</h4>
                    <div className="max-h-96 overflow-y-auto space-y-2">
                      {analysisResult.detected_dents.map((dent, index) => (
                        <div key={index} className="border rounded-lg p-3 space-y-2">
                          <div className="flex items-center justify-between">
                            <Badge className={getSeverityColor(dent.severity)}>
                              {dent.severity}
                            </Badge>
                            <span className="font-medium">${dent.estimated_cost.toFixed(0)}</span>
                          </div>
                          <div className="grid grid-cols-2 gap-2 text-sm text-gray-600">
                            <div>Location: {dent.location}</div>
                            <div>Size: {dent.size}</div>
                            <div>Panel: {dent.panel}</div>
                            <div>Confidence: {(dent.confidence * 100).toFixed(1)}%</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <Separator />

                  <div className="space-y-2">
                    <h4 className="font-medium text-gray-900">Detailed Report</h4>
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <pre className="text-sm whitespace-pre-wrap font-mono">
                        {analysisResult.detailed_report}
                      </pre>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="image" className="space-y-4">
                  {analysisResult.processed_image && (
                    <div className="space-y-4">
                      <h4 className="font-medium text-gray-900">Processed Image with Annotations</h4>
                      <div className="border rounded-lg overflow-hidden">
                        <img
                          src={analysisResult.processed_image}
                          alt="Processed vehicle with damage annotations"
                          className="w-full h-auto"
                        />
                      </div>
                      <p className="text-sm text-gray-600">
                        Detected dents are highlighted with colored circles. Green = Minor, Orange = Moderate, Red = Severe.
                      </p>
                    </div>
                  )}
                </TabsContent>
              </Tabs>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Action Buttons */}
      {analysisResult && (
        <Card>
          <CardContent className="pt-6">
            <div className="flex flex-wrap gap-4 justify-center">
              <Button variant="outline" onClick={() => {
                const report = analysisResult.detailed_report;
                const blob = new Blob([report], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'hail-damage-report.txt';
                a.click();
                URL.revokeObjectURL(url);
              }}>
                Download Report
              </Button>
              
              <Button variant="outline" onClick={() => {
                const data = JSON.stringify(analysisResult, null, 2);
                const blob = new Blob([data], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'hail-damage-data.json';
                a.click();
                URL.revokeObjectURL(url);
              }}>
                Export Data
              </Button>
              
              <Button onClick={() => {
                setAnalysisResult(null);
                setSelectedFile(null);
                setPreviewUrl(null);
                setError(null);
              }}>
                New Analysis
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default HailDamageAnalyzer;
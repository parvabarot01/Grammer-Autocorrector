export type PublicChange = {
  before: string;
  after: string;
  explanation: string;
};

export type PublicCorrectionResponse = {
  original_text: string;
  corrected_text: string;
  changes: PublicChange[];
  summary: string;
  success: boolean;
};

export type PublicCorrectionRequest = {
  text: string;
};

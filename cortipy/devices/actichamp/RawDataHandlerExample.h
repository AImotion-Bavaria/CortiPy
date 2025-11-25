#pragma once
#ifndef _RAW_DATA_HANDLER_
#define _RAW_DATA_HANDLER_

#include "SDK.h"
#include "AmplifierSDK.h"
#include <vector>

class RawDataHandler
{
private:
	std::vector<int> m_vnChannelTypes;
	std::vector<float> m_vfChannelResolutions;
	std::vector<int> m_vnChannelDataTypes;
	int m_nSampleSize;
	int m_nEnabledChannelCnt;
	int m_nSamplingRate;
	int m_nReadBlockLength;
	vector<BYTE> m_Buffer;

	/**
	* determine the sample size of the incoming data stream
	*/
	void IncrementSampleSize(int nDataType)
	{
		switch (nDataType)
		{
		case DT_INT16:
		case DT_UINT16:
		{
			m_nSampleSize += 2;
		}
		break;
		case DT_INT32:
		case DT_UINT32:
		case DT_FLOAT32:
		{
			m_nSampleSize += 4;

		}
		break;
		case DT_INT64:
		case DT_UINT64:
		case DT_FLOAT64:
		{
			m_nSampleSize += 8;
		}
		break;
		default:
			break;
		}
	}

public:
	int getSampleSize() { return m_nSampleSize; }

	int getEnabledChannelCount() const { 
        return m_nEnabledChannelCnt; 
    }

	RawDataHandler(CAmplifier& amp)
	{

		int nAvailableChannels = 0;
		int nIsEnabled = 0;
		int nChannelType;
		float fResolution;
		int nDataType;
		int nRet;
		float fBaseSamplingRate;
		float fSubSampleDivisor;
		std::string sChannelFunc;

		m_vfChannelResolutions.clear();
		m_vnChannelTypes.clear();
		m_vnChannelDataTypes.clear();
		m_nSampleSize = 0;
		m_nEnabledChannelCnt = 0;
		m_nSampleSize = 0;

		nRet = amp.GetProperty(fBaseSamplingRate, DPROP_F32_BaseSampleRate);
		nRet = amp.GetProperty(fSubSampleDivisor, DPROP_F32_SubSampleDivisor);
		m_nSamplingRate = fBaseSamplingRate / fSubSampleDivisor;

		amp.GetProperty(nAvailableChannels, DPROP_I32_AvailableChannels);
		amp.StartAcquisition(RM_NORMAL);
		m_nSampleSize += sizeof(UINT64);
		for (int i = 0; i < nAvailableChannels; i++)
		{
			nIsEnabled = 0;
			amp.GetProperty(nIsEnabled, i, CPROP_B32_RecordingEnabled);
			amp.GetProperty(nChannelType, i, CPROP_I32_Type);
			amp.GetProperty(sChannelFunc, i, CPROP_CHR_Function);
			if (nIsEnabled == 1)
			{
				m_vnChannelTypes.push_back(nChannelType);

				amp.GetProperty(fResolution, i, CPROP_F32_Resolution);
				m_vfChannelResolutions.push_back(fResolution);

				amp.GetProperty(nDataType, i, CPROP_I32_DataType);
				IncrementSampleSize(nDataType);
				m_vnChannelDataTypes.push_back(nDataType);
				m_nEnabledChannelCnt++;
			}
		}
		//This defines a read length of 200 ms per GetData 
		m_nReadBlockLength = int(m_nSampleSize * m_nSamplingRate / 5);
		m_Buffer.resize(m_nReadBlockLength, 0);
		amp.StopAcquisition();
	}


	int ParseRawData(CAmplifier& amp, std::vector<std::vector<float>>& vvfDataOut)
	{
		uint64_t sc;

		int nOffset = 0;
		float fSample = 0;
		int mTriggTmp = 0;
		int nIsSecondBit = 0;
		int nRet = -1;

		//while (nRet <= 0)
		nRet = amp.GetData(&m_Buffer[0], (int)m_Buffer.size(), (int)m_Buffer.size() / m_nSampleSize);
		if (nRet <= 0)return nRet;
		int64_t nNumSamples = nRet / m_nSampleSize;

		std::vector<float> vfTmpData;
		vvfDataOut.clear();
		for (int s = 0; s < nNumSamples; s++)
		{
			nOffset = 0;
			sc = *(uint64_t*)&m_Buffer[s * m_nSampleSize + nOffset];
			nOffset += 8; // sample counter offset 

			vfTmpData.resize(m_nEnabledChannelCnt);

			for (int i = 0; i < m_nEnabledChannelCnt; i++)
			{
				switch (m_vnChannelDataTypes[i])
				{
				case DT_INT16:
				{
					int16_t tmp = *(int16_t*)&m_Buffer[s * m_nSampleSize + nOffset];
					fSample = (float)tmp * m_vfChannelResolutions[i];
					nOffset += 2;
					break;
				}
				case DT_UINT16:
				{

					uint16_t tmp = *(uint16_t*)&m_Buffer[s * m_nSampleSize + nOffset];
					fSample = (float)tmp * m_vfChannelResolutions[i];
					nOffset += 2;
					break;
				}
				case DT_INT32:
				{
					int32_t tmp = *(int32_t*)&m_Buffer[s * m_nSampleSize + nOffset];
					fSample = (float)(tmp)*m_vfChannelResolutions[i];
					nOffset += 4;
					break;
				}
				case DT_UINT32:
				{
					uint32_t tmp = *(uint32_t*)&m_Buffer[s * m_nSampleSize + nOffset];
					fSample = (float)tmp * m_vfChannelResolutions[i];
					nOffset += 4;
					break;
				}
				case DT_FLOAT32:
				{
					float tmp = *(float*)&m_Buffer[s * m_nSampleSize + nOffset];
					fSample = (float)tmp * m_vfChannelResolutions[i];
					nOffset += 4;
					break;
				}
				case DT_INT64:
				{
					int64_t tmp = *(int64_t*)&m_Buffer[s * m_nSampleSize + nOffset];
					fSample = (float)tmp * m_vfChannelResolutions[i];
					nOffset += 8;
					break;
				}
				case DT_UINT64:
				{
					uint64_t tmp = *(uint64_t*)&m_Buffer[s * m_nSampleSize + nOffset];
					fSample = (float)tmp * m_vfChannelResolutions[i];
					nOffset += 8;
					break;
				}
				case DT_FLOAT64:
				{

					float tmp = *(float*)&m_Buffer[s * m_nSampleSize + nOffset];
					fSample = (float)tmp * m_vfChannelResolutions[i];
					nOffset += 8;
					break;
				}
				default:
					break;
				}

				vfTmpData[i] = fSample;
			}
			vvfDataOut.push_back(vfTmpData);
		}
		return nRet / m_nSampleSize;
	}

};

#endif

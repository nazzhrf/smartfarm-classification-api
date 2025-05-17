import {
  Card,
  CardHeader,
  Grid,
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress,
} from '@mui/material';
import { useEffect, useState } from 'react';
import axios from 'axios';

export default function AppSeverityLog() {
  const trayImagePath = '/assets/images/Tray/chili_detection_order.jpg';
  const fallbackImage = '/assets/images/fallback/sample-disease.png';

  const [imageSrc, setImageSrc] = useState(trayImagePath);
  const [logData, setLogData] = useState([]);
  const [loading, setLoading] = useState(true);

  const handleImageError = () => {
    setImageSrc(fallbackImage);
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await axios.post('https://api-classify.smartfarm.id/get-data', {});
        setLogData(res.data);
        console.log(res.data);
      } catch (error) {
        console.error('Failed to fetch data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  return (
    <Card sx={{ p: 3 }}>
      <CardHeader title="Severity Level Log" />
      <Grid container spacing={3}>
        {/* Table section */}
        <Grid item xs={12} md={6}>
          <TableContainer component={Paper}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Image</TableCell>
                  <TableCell>Date</TableCell>
                  <TableCell>Time</TableCell>
                  <TableCell>Pred Class 1</TableCell>
                  <TableCell>Pred Class 2</TableCell>
                  <TableCell>Pred Class 3</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {loading ? (
                  <TableRow>
                    <TableCell colSpan={6} align="center">
                      <CircularProgress size={24} />
                    </TableCell>
                  </TableRow>
                ) : (
                  logData.map((item, index) => (
                    <TableRow key={index}>
                      <TableCell>{item.image}</TableCell>
                      <TableCell>{item.tanggal}</TableCell>
                      <TableCell>{item.waktu}</TableCell>
                      <TableCell>{item.pred_class_1}</TableCell>
                      <TableCell>{item.pred_class_2}</TableCell>
                      <TableCell>{item.pred_class_3}</TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </Grid>

        {/* Image viewer section */}
        <Grid item xs={12} md={6}>
          <Box
            sx={{
              borderRadius: 2,
              padding: 1,
              backgroundColor: '#FFFFFF',
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              height: 400,
            }}
          >
            <Box
              component="img"
              src={imageSrc}
              alt="Tray Photo not Found"
              onError={handleImageError}
              sx={{
                maxHeight: '100%',
                maxWidth: '100%',
                objectFit: 'contain',
                borderRadius: 1,
              }}
            />
          </Box>
        </Grid>
      </Grid>
    </Card>
  );
}
